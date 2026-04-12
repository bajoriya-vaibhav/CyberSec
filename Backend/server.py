"""
FastAPI REST server for real-time deepfake detection.
Receives JPEG frames + WAV audio from Android app, returns predictions.
Designed for near real-time inference with minimal latency.

Usage:
    python server.py
    # or: uvicorn server:app --host 0.0.0.0 --port 7860 --reload
"""

import os
import sys
import time
import json
import tempfile
import logging
import asyncio
import threading
from io import BytesIO
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image

from detector import DeepFakeDetector
from config import Config

# ─── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO if Config.VERBOSE else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─── Global detector (loaded once) ──────────────────────────────────
detector: Optional[DeepFakeDetector] = None
model_load_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global detector, model_load_time
    logger.info("=" * 60)
    logger.info("Starting DeepFake Detection Server...")
    logger.info("=" * 60)

    t0 = time.time()
    try:
        detector = DeepFakeDetector()
        model_load_time = time.time() - t0
        logger.info(f"Models loaded in {model_load_time:.1f}s")
        logger.info("Server ready for predictions!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        raise RuntimeError(f"Model loading failed: {e}")

    yield

    # Cleanup
    logger.info("Server shutting down...")
    detector = None


# ─── FastAPI App ─────────────────────────────────────────────────────
app = FastAPI(
    title="DeepFake Detection API",
    description="Real-time deepfake detection from video frames and audio",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Stats tracking ─────────────────────────────────────────────────
request_count = 0
total_inference_time = 0.0
inference_lock = threading.Lock()  # Prevent concurrent model inference


# ─── Synchronous inference helper (runs in thread pool) ──────────────

def _run_ws_inference(frame_jpeg_list: list, audio_wav_bytes: bytes = None) -> dict:
    """
    Run GenConViT batch inference + optional audio analysis.
    Called from asyncio thread pool — must be thread-safe.
    """
    global request_count, total_inference_time

    with inference_lock:
        t_start = time.time()
        request_count += 1
        req_id = request_count

        logger.info(f"[WS #{req_id}] Inference: {len(frame_jpeg_list)} frames, "
                    f"audio={'yes' if audio_wav_bytes else 'no'}")

        video_score = None
        audio_score = None
        security_alert = None
        threat_vector = None

        # ── Decode frames ──
        all_images = []
        for i, jpeg_bytes in enumerate(frame_jpeg_list):
            try:
                img = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
                all_images.append(img)
            except Exception as e:
                logger.warning(f"[WS #{req_id}] Frame {i} decode error: {e}")

        # Subsample to max 15 frames (detection is expensive per-frame)
        MAX_FRAMES = 15
        if len(all_images) > MAX_FRAMES:
            step = len(all_images) / MAX_FRAMES
            images = [all_images[int(i * step)] for i in range(MAX_FRAMES)]
            logger.info(f"[WS #{req_id}] Subsampled {len(all_images)} → {len(images)} frames")
        else:
            images = all_images

        person_detected = True

        if images:
            try:
                video_result = detector.video_detector.detect(images)
                video_score = video_result.fake_probability
                person_detected = video_result.metadata.get('person_detected', True)
                logger.info(f"[WS #{req_id}] Video: fake={video_score:.4f}, "
                           f"person={person_detected}, "
                           f"regions={video_result.metadata.get('regions_found', '?')}/{len(images)}")
            except Exception as e:
                logger.error(f"[WS #{req_id}] Video inference error: {e}")

        # ── Audio analysis ──
        if audio_wav_bytes and len(audio_wav_bytes) > 200:
            tmp_audio = None
            try:
                tmp_audio = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="ws_audio_"
                )
                tmp_audio.write(audio_wav_bytes)
                tmp_audio.flush()
                tmp_audio.close()
                audio_result = detector.analyze_audio(tmp_audio.name)
                audio_score = audio_result.get('fake_probability', 0.5)
                logger.info(f"[WS #{req_id}] Audio: fake={audio_score:.4f}")
            except Exception as e:
                logger.error(f"[WS #{req_id}] Audio inference error: {e}")
            finally:
                if tmp_audio:
                    try: os.unlink(tmp_audio.name)
                    except OSError: pass

        # ── Fusion ──
        if video_score is not None and audio_score is not None:
            from detectors.base_detector import DetectionResult as DR
            v_r = DR(fake_probability=video_score, real_probability=1.0 - video_score)
            a_r = DR(fake_probability=audio_score, real_probability=1.0 - audio_score)
            fused = detector.fusion_strategy.fuse(v_r, a_r)
            final_fake = fused.fake_probability
            final_confidence = fused.confidence
            security_alert = fused.metadata.get('security_alert')
            threat_vector = fused.metadata.get('threat_vector')
            prediction = "Suspicious" if security_alert == 'MISMATCH_DETECTED' else \
                         ("Fake" if final_fake > 0.5 else "Real")
        elif video_score is not None:
            final_fake = video_score
            final_confidence = max(video_score, 1.0 - video_score)
            prediction = "Fake" if video_score > 0.5 else "Real"
        elif audio_score is not None:
            final_fake = audio_score
            final_confidence = max(audio_score, 1.0 - audio_score)
            prediction = "Fake" if audio_score > 0.5 else "Real"
        else:
            prediction = "Unknown"
            final_confidence = 0.0
            final_fake = 0.5

        inference_time = time.time() - t_start
        total_inference_time += inference_time

        rl_weights = None
        if hasattr(detector.fusion_strategy, 'video_weight'):
            rl_weights = {
                'video_weight': round(float(detector.fusion_strategy.video_weight), 3),
                'audio_weight': round(float(detector.fusion_strategy.audio_weight), 3),
            }

        result = {
            "type": "result",
            "prediction": prediction,
            "confidence": round(float(final_confidence), 4),
            "fake_probability": round(float(final_fake), 4),
            "video_fake_score": round(float(video_score), 4) if video_score is not None else None,
            "audio_fake_score": round(float(audio_score), 4) if audio_score is not None else None,
            "person_detected": person_detected,
            "security_alert": security_alert,
            "threat_vector": threat_vector,
            "analysis_source": "local",
            "fusion_mode": Config.FUSION_MODE,
            "rl_weights": rl_weights,
            "inference_time_ms": round(inference_time * 1000, 1),
            "frames_analyzed": len(images),
        }

        logger.info(f"[WS #{req_id}] Result: {prediction} "
                    f"(conf={final_confidence:.3f}, person={person_detected}, "
                    f"frames={len(images)}, time={inference_time*1000:.0f}ms)")

        return result


# ─── WebSocket Streaming Endpoint ────────────────────────────────────

@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    """
    WebSocket endpoint for continuous streaming deepfake analysis.
    
    Protocol:
      Client → Server (binary):  0x01 + JPEG bytes  (video frame)
      Client → Server (binary):  0x02 + WAV bytes   (audio chunk)
      Client → Server (text):    {"type":"config", "duration":15, "audio_enabled":false}
      Server → Client (text):    {"type":"result", "prediction":"Fake", ...}
      Server → Client (text):    {"type":"status", "frames_buffered":12, ...}
      Server → Client (text):    {"type":"analyzing", "frames":15}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    if detector is None:
        await websocket.send_json({"type": "error", "message": "Models not loaded"})
        await websocket.close()
        return

    # Shared state
    frame_buffer: list = []
    audio_data: bytearray = bytearray()
    config = {"duration": 15, "audio_enabled": False}
    buffer_lock = asyncio.Lock()
    send_lock = asyncio.Lock()
    running = True
    last_analysis_time = time.time()

    async def safe_send(data: dict):
        """Thread-safe JSON send."""
        try:
            async with send_lock:
                await websocket.send_json(data)
        except Exception:
            pass

    # ── Task 1: Receive frames (never blocks) ──
    async def receiver():
        nonlocal running
        try:
            while running:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    running = False
                    break

                if "text" in message:
                    try:
                        ctrl = json.loads(message["text"])
                        if ctrl.get("type") == "config":
                            config["duration"] = ctrl.get("duration", 15)
                            config["audio_enabled"] = ctrl.get("audio_enabled", False)
                            logger.info(f"WS config: duration={config['duration']}s, "
                                       f"audio={config['audio_enabled']}")
                    except json.JSONDecodeError:
                        pass

                elif "bytes" in message:
                    raw = message["bytes"]
                    if len(raw) < 2:
                        continue
                    marker = raw[0]
                    payload = raw[1:]
                    async with buffer_lock:
                        if marker == 0x01:  # Video frame
                            frame_buffer.append(bytes(payload))
                            # Cap buffer to prevent memory overflow
                            if len(frame_buffer) > 200:
                                frame_buffer.pop(0)
                        elif marker == 0x02:  # Audio chunk
                            audio_data.clear()
                            audio_data.extend(payload)  # Keep latest audio
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(f"WS receiver error: {e}")
        finally:
            running = False

    # ── Task 2: Periodic analysis (runs inference in thread pool) ──
    async def analyzer():
        nonlocal running, last_analysis_time
        try:
            while running:
                await asyncio.sleep(1)  # Check every second
                if not running:
                    break

                elapsed = time.time() - last_analysis_time
                if elapsed < config["duration"]:
                    continue

                # Grab buffered frames
                async with buffer_lock:
                    if not frame_buffer:
                        last_analysis_time = time.time()
                        continue
                    frames_copy = list(frame_buffer)
                    audio_copy = bytes(audio_data) if config["audio_enabled"] and audio_data else None
                    frame_buffer.clear()
                    audio_data.clear()

                # Run inference in thread pool (non-blocking)
                try:
                    result = await asyncio.to_thread(
                        _run_ws_inference, frames_copy, audio_copy
                    )
                    if running:
                        await safe_send(result)
                except Exception as e:
                    logger.error(f"WS inference error: {e}")
                    await safe_send({"type": "error", "message": str(e)})

                last_analysis_time = time.time()
        except Exception as e:
            logger.warning(f"WS analyzer error: {e}")
        finally:
            running = False

    # ── Task 3: Status updates (so client knows we're alive) ──
    async def status_sender():
        try:
            while running:
                await asyncio.sleep(2)
                if not running:
                    break
                async with buffer_lock:
                    n = len(frame_buffer)
                elapsed = time.time() - last_analysis_time
                remaining = max(0, config["duration"] - elapsed)
                await safe_send({
                    "type": "status",
                    "frames_buffered": n,
                    "seconds_until_analysis": round(remaining, 0),
                })
        except Exception:
            pass

    # Run all three tasks concurrently
    try:
        await asyncio.gather(receiver(), analyzer(), status_sender())
    except Exception as e:
        logger.warning(f"WS session ended: {e}")
    finally:
        logger.info("WebSocket client disconnected")


# ─── Health / Status ─────────────────────────────────────────────────

@app.get("/")
async def root():
    """Server status page."""
    return {
        "status": "running",
        "service": "DeepFake Detection API",
        "version": "3.1.0",
        "models_loaded": detector is not None,
        "model_load_time_seconds": round(model_load_time, 2),
        "video_model": f"GenConViT-{Config.CVIT_NET}",
        "audio_model": Config.AUDIO_MODEL,
        "fusion_mode": Config.FUSION_MODE,
    }


@app.get("/health")
async def health():
    """Health check endpoint for the Android app."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ok", "models_loaded": True}


@app.post("/check_person")
async def check_person(frame: UploadFile = File(...)):
    """
    Lightweight person/face presence check.
    Used by Android app for auto-start/stop scanning.
    No model inference — only OpenCV Haar + HOG detection.
    """
    if detector is None or detector.video_detector is None:
        raise HTTPException(status_code=503, detail="Detector not loaded")
    
    try:
        data = await frame.read()
        img = Image.open(BytesIO(data)).convert("RGB")
        result = detector.video_detector.has_person(img)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"check_person error: {e}")
        return JSONResponse(content={"person_detected": False, "type": "error", "count": 0})


# ─── File Upload Analysis Endpoints ──────────────────────────────────

_BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
_VERIFIED_DIR = os.path.join(_BACKEND_ROOT, "verified")
_REAL_DIR = os.path.join(_VERIFIED_DIR, "real")
_FAKE_DIR = os.path.join(_VERIFIED_DIR, "fake")


def _ensure_verified_dirs():
    os.makedirs(_REAL_DIR, exist_ok=True)
    os.makedirs(_FAKE_DIR, exist_ok=True)


def _extract_video_frames_sync(video_bytes: bytes, max_frames: int = 15) -> list:
    """Extract up to max_frames evenly spaced PIL frames from a video in memory."""
    import cv2
    import numpy as np

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_bytes)
    tmp.flush()
    tmp.close()
    frames = []
    try:
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            step = max(1, total // max_frames)
            idx = 0
            while len(frames) < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                idx += step
        cap.release()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return frames


@app.post("/analyze_file")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyze an uploaded image or video for deepfake detection.

    Accepts:
        Image (JPEG, PNG, WEBP) → single frame analysis
        Video (MP4, MOV, AVI, MKV) → extracts 15 evenly-spaced frames

    Returns:
        prediction, confidence, fake_probability, frames_analyzed,
        person_detected, file_type, inference_time_ms
    """
    if detector is None or detector.video_detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    t_start = time.time()
    data = await file.read()
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    ext = os.path.splitext(filename)[1]

    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
    is_video = ext in VIDEO_EXTS or "video" in content_type

    file_type = "video" if is_video else "image"
    frames = []

    try:
        if is_video:
            frames = await asyncio.to_thread(_extract_video_frames_sync, data)
            if not frames:
                raise ValueError("No frames extracted — is this a valid video?")
        else:
            frames = [Image.open(BytesIO(data)).convert("RGB")]
    except Exception as e:
        logger.error(f"[FILE] Decode error ({filename}): {e}")
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    logger.info(f"[FILE] Analyzing {file_type}: {filename} ({len(frames)} frames)")

    def _run():
        with inference_lock:
            result = detector.video_detector.detect(frames)
            return result

    result = await asyncio.to_thread(_run)
    fake_prob = result.fake_probability
    real_prob = 1.0 - fake_prob
    prediction = "Fake" if fake_prob > 0.5 else "Real"
    confidence = max(fake_prob, real_prob)
    person_detected = result.metadata.get("person_detected", True)
    inference_time = time.time() - t_start

    logger.info(
        f"[FILE] {prediction} fake={fake_prob:.4f} "
        f"conf={confidence:.4f} frames={len(frames)} "
        f"time={inference_time*1000:.0f}ms"
    )

    return JSONResponse(content={
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "fake_probability": round(float(fake_prob), 4),
        "real_probability": round(float(real_prob), 4),
        "person_detected": person_detected,
        "frames_analyzed": len(frames),
        "file_type": file_type,
        "filename": file.filename,
        "inference_time_ms": round(inference_time * 1000, 1),
    })


@app.post("/verify")
async def verify_file(
    file: UploadFile = File(...),
    label: str = Form(...),
):
    """
    Save a user-verified file to verified/real/ or verified/fake/ folder.
    Only user-submitted labels are saved — no live streams or frames.

    Args:
        file: Image or video file
        label: "real" or "fake"
    """
    label = label.strip().lower()
    if label not in ("real", "fake"):
        raise HTTPException(status_code=400, detail="label must be 'real' or 'fake'")

    _ensure_verified_dirs()
    dest_dir = _REAL_DIR if label == "real" else _FAKE_DIR

    # Sanitize filename + avoid collisions
    raw_name = file.filename or f"file_{int(time.time())}"
    safe_name = "".join(c for c in raw_name if c.isalnum() or c in "._-")
    base, ext = os.path.splitext(safe_name)
    dest_path = os.path.join(dest_dir, safe_name)
    counter = 1
    while os.path.exists(dest_path):
        dest_path = os.path.join(dest_dir, f"{base}_{counter}{ext}")
        counter += 1

    data = await file.read()
    with open(dest_path, "wb") as f:
        f.write(data)

    relative = os.path.relpath(dest_path, _BACKEND_ROOT).replace("\\", "/")
    logger.info(f"[VERIFY] Saved {label}: {relative} ({len(data)} bytes)")

    return JSONResponse(content={
        "saved": True,
        "label": label,
        "filename": os.path.basename(dest_path),
        "relative_path": relative,
        "size_bytes": len(data),
    })


@app.get("/stats")
async def stats():
    """Server statistics."""
    avg_inference = (total_inference_time / request_count) if request_count > 0 else 0
    rl_stats = {}
    if hasattr(detector, 'fusion_strategy') and hasattr(detector.fusion_strategy, 'get_performance_stats'):
        rl_stats = detector.fusion_strategy.get_performance_stats()

    return {
        "total_requests": request_count,
        "average_inference_ms": round(avg_inference * 1000, 1),
        "fusion_mode": Config.FUSION_MODE,
        "rl_stats": rl_stats,
    }


# ─── Batch Prediction Endpoint (primary — used by Android app) ────────

@app.post("/predict_batch")
async def predict_batch(
    video_frames: List[UploadFile] = File(default=[]),
    audio_segment: Optional[UploadFile] = File(None),
):
    """
    Analyze a BATCH of video frames and optional audio for deepfake detection.
    
    This is the primary endpoint for the Android app. It sends all frames
    accumulated during one analysis cycle at once, allowing GenConViT to
    analyze them as a batch (how the model was designed to work).
    
    Accepts multipart/form-data with:
    - video_frames: Multiple JPEG images
    - audio_segment: WAV audio (optional)
    
    Returns a single cohesive prediction.
    """
    global request_count, total_inference_time

    if detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    if len(video_frames) == 0 and audio_segment is None:
        raise HTTPException(status_code=400, detail="No frames or audio provided")

    t_start = time.time()
    request_count += 1
    req_id = request_count

    logger.info(f"[BATCH #{req_id}] Batch request: "
                f"frames={len(video_frames)}, "
                f"audio={'yes' if audio_segment else 'no'}")

    video_score = None
    audio_score = None
    security_alert = None
    threat_vector = None
    analysis_source = "local"

    try:
        # ── Analyze video frames as batch ────────────────────────
        if len(video_frames) > 0:
            images = []
            for i, frame_file in enumerate(video_frames):
                try:
                    frame_bytes = await frame_file.read()
                    img = Image.open(BytesIO(frame_bytes)).convert("RGB")
                    images.append(img)
                except Exception as e:
                    logger.warning(f"[BATCH #{req_id}] Frame {i} decode failed: {e}")
            
            if images:
                logger.info(f"[BATCH #{req_id}] Decoded {len(images)} frames, running batch inference...")
                
                # Use multi-frame detect (this is how GenConViT was designed)
                from detectors.base_detector import DetectionResult as DR
                video_result = detector.video_detector.detect(images)
                video_score = video_result.fake_probability
                analysis_source = video_result.metadata.get('source', 'cvit')
                
                logger.info(f"[BATCH #{req_id}] Video batch result: "
                           f"fake={video_score:.4f}, "
                           f"faces={video_result.metadata.get('faces_found', '?')}/{len(images)} frames")

        # ── Analyze audio segment ────────────────────────────────
        if audio_segment is not None:
            audio_bytes = await audio_segment.read()
            logger.info(f"[BATCH #{req_id}] Audio size: {len(audio_bytes)} bytes")

            if len(audio_bytes) > 100:
                tmp_audio = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="df_audio_"
                )
                try:
                    tmp_audio.write(audio_bytes)
                    tmp_audio.flush()
                    tmp_audio.close()

                    audio_result = detector.analyze_audio(tmp_audio.name)
                    audio_score = audio_result.get('fake_probability', 0.5)
                finally:
                    try:
                        os.unlink(tmp_audio.name)
                    except OSError:
                        pass

        # ── Fusion ───────────────────────────────────────────────
        if video_score is not None and audio_score is not None:
            from detectors.base_detector import DetectionResult as DR
            v_result = DR(fake_probability=video_score, real_probability=1.0 - video_score)
            a_result = DR(fake_probability=audio_score, real_probability=1.0 - audio_score)
            fused = detector.fusion_strategy.fuse(v_result, a_result)
            final_fake = fused.fake_probability
            final_confidence = fused.confidence
            security_alert = fused.metadata.get('security_alert')
            threat_vector = fused.metadata.get('threat_vector')
            prediction = "Suspicious" if security_alert == 'MISMATCH_DETECTED' else \
                         ("Fake" if final_fake > 0.5 else "Real")
        elif video_score is not None:
            final_fake = video_score
            final_confidence = max(video_score, 1.0 - video_score)
            prediction = "Fake" if video_score > 0.5 else "Real"
        elif audio_score is not None:
            final_fake = audio_score
            final_confidence = max(audio_score, 1.0 - audio_score)
            prediction = "Fake" if audio_score > 0.5 else "Real"
        else:
            prediction = "Unknown"
            final_confidence = 0.0
            final_fake = 0.5

        # ── Response ─────────────────────────────────────────────
        inference_time = time.time() - t_start
        total_inference_time += inference_time

        rl_weights = None
        if hasattr(detector.fusion_strategy, 'video_weight'):
            rl_weights = {
                'video_weight': round(float(detector.fusion_strategy.video_weight), 3),
                'audio_weight': round(float(detector.fusion_strategy.audio_weight), 3),
            }

        response = {
            "prediction": prediction,
            "confidence": round(float(final_confidence), 4),
            "fake_probability": round(float(final_fake), 4),
            "video_fake_score": round(float(video_score), 4) if video_score is not None else None,
            "audio_fake_score": round(float(audio_score), 4) if audio_score is not None else None,
            "security_alert": security_alert,
            "threat_vector": threat_vector,
            "analysis_source": analysis_source,
            "fusion_mode": Config.FUSION_MODE,
            "rl_weights": rl_weights,
            "inference_time_ms": round(inference_time * 1000, 1),
            "frames_analyzed": len(video_frames),
        }

        logger.info(f"[BATCH #{req_id}] Result: {prediction} "
                     f"(confidence={final_confidence:.3f}, "
                     f"frames={len(video_frames)}, "
                     f"time={inference_time*1000:.0f}ms)")

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[BATCH #{req_id}] Inference error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"prediction": "Error", "confidence": 0.0, "error": str(e)}
        )


# ─── Single-Frame Prediction Endpoint (legacy) ──────────────────────

@app.post("/predict")
async def predict(
    video_frame: Optional[UploadFile] = File(None),
    audio_segment: Optional[UploadFile] = File(None),
):
    """
    Analyze a video frame and/or audio segment for deepfake detection.

    Accepts multipart/form-data with:
    - video_frame: JPEG image (optional)
    - audio_segment: WAV audio (optional)

    Returns:
    - prediction: "Real" or "Fake"
    - confidence: 0.0 to 1.0
    - details: extended analysis info
    """
    global request_count, total_inference_time

    if detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    if video_frame is None and audio_segment is None:
        raise HTTPException(status_code=400, detail="No video_frame or audio_segment provided")

    t_start = time.time()
    request_count += 1
    req_id = request_count

    logger.info(f"[REQ #{req_id}] Prediction request: "
                f"frame={'yes' if video_frame else 'no'}, "
                f"audio={'yes' if audio_segment else 'no'}")

    result = {}
    video_score = None
    audio_score = None
    security_alert = None
    threat_vector = None
    analysis_source = "local"

    try:
        # ── Analyze video frame ──────────────────────────────────
        if video_frame is not None:
            frame_bytes = await video_frame.read()
            logger.info(f"[REQ #{req_id}] Frame size: {len(frame_bytes)} bytes")

            try:
                image = Image.open(BytesIO(frame_bytes)).convert("RGB")
            except Exception as e:
                logger.error(f"[REQ #{req_id}] Failed to decode frame: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

            # Run single-frame inference (fast path)
            image_result = detector.analyze_image(image)
            video_score = image_result.get('fake_probability', 0.5)
            analysis_source = image_result.get('analysis_source', 'local')

            result.update({
                'video_verdict': image_result['verdict'],
                'video_confidence': image_result['confidence'],
                'video_fake_score': video_score,
            })

        # ── Analyze audio segment ────────────────────────────────
        if audio_segment is not None:
            audio_bytes = await audio_segment.read()
            logger.info(f"[REQ #{req_id}] Audio size: {len(audio_bytes)} bytes")

            if len(audio_bytes) > 100:  # Skip tiny/empty audio
                # Write to temp file for pipeline compatibility
                tmp_audio = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="df_audio_"
                )
                try:
                    tmp_audio.write(audio_bytes)
                    tmp_audio.flush()
                    tmp_audio.close()

                    audio_result = detector.analyze_audio(tmp_audio.name)
                    audio_score = audio_result.get('fake_probability', 0.5)

                    result.update({
                        'audio_verdict': audio_result['verdict'],
                        'audio_confidence': audio_result['confidence'],
                        'audio_fake_score': audio_score,
                    })
                finally:
                    # Always cleanup temp file
                    try:
                        os.unlink(tmp_audio.name)
                    except OSError:
                        pass
            else:
                logger.info(f"[REQ #{req_id}] Audio segment too small, skipping")

        # ── Fusion logic ─────────────────────────────────────────
        if video_score is not None and audio_score is not None:
            # Both modalities available — use fusion strategy
            from detectors.base_detector import DetectionResult as DR

            v_result = DR(fake_probability=video_score, real_probability=1.0 - video_score)
            a_result = DR(fake_probability=audio_score, real_probability=1.0 - audio_score)

            fused = detector.fusion_strategy.fuse(v_result, a_result)

            final_fake = fused.fake_probability
            final_confidence = fused.confidence

            # Extract security info
            security_alert = fused.metadata.get('security_alert')
            threat_vector = fused.metadata.get('threat_vector')

            # Override verdict if suspicious mismatch
            if security_alert == 'MISMATCH_DETECTED':
                prediction = "Suspicious"
            else:
                prediction = "Fake" if final_fake > 0.5 else "Real"

        elif video_score is not None:
            # Video only
            final_fake = video_score
            final_confidence = max(video_score, 1.0 - video_score)
            prediction = "Fake" if video_score > 0.5 else "Real"

        elif audio_score is not None:
            # Audio only
            final_fake = audio_score
            final_confidence = max(audio_score, 1.0 - audio_score)
            prediction = "Fake" if audio_score > 0.5 else "Real"

        else:
            prediction = "Unknown"
            final_confidence = 0.0
            final_fake = 0.5

        # ── Build response ───────────────────────────────────────
        inference_time = time.time() - t_start
        total_inference_time += inference_time

        # Get RL weights if available
        rl_weights = None
        if hasattr(detector.fusion_strategy, 'video_weight'):
            rl_weights = {
                'video_weight': round(float(detector.fusion_strategy.video_weight), 3),
                'audio_weight': round(float(detector.fusion_strategy.audio_weight), 3),
            }

        response = {
            # Primary fields (Android app expects these)
            "prediction": prediction,
            "confidence": round(float(final_confidence), 4),

            # Extended details
            "fake_probability": round(float(final_fake), 4),
            "video_fake_score": round(float(video_score), 4) if video_score is not None else None,
            "audio_fake_score": round(float(audio_score), 4) if audio_score is not None else None,
            "security_alert": security_alert,
            "threat_vector": threat_vector,
            "analysis_source": analysis_source,
            "fusion_mode": Config.FUSION_MODE,
            "rl_weights": rl_weights,
            "inference_time_ms": round(inference_time * 1000, 1),
        }

        logger.info(f"[REQ #{req_id}] Result: {prediction} "
                     f"(confidence={final_confidence:.3f}, "
                     f"time={inference_time*1000:.0f}ms)")

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[REQ #{req_id}] Inference error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "prediction": "Error",
                "confidence": 0.0,
                "error": str(e),
            }
        )


# ─── Feedback Endpoint (for RL learning) ─────────────────────────────

@app.post("/feedback")
async def submit_feedback(
    correct_label: str = Form(...),
    user_confidence: float = Form(1.0),
    video_fake_score: Optional[float] = Form(None),
    audio_fake_score: Optional[float] = Form(None),
    original_verdict: Optional[str] = Form(None),
):
    """
    Submit user feedback for RL weight learning.

    Args:
        correct_label: "REAL", "FAKE", or "SUSPICIOUS"
        user_confidence: 0.0 to 1.0
        video_fake_score: Original video score from prediction
        audio_fake_score: Original audio score from prediction
        original_verdict: Original prediction verdict
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if Config.FUSION_MODE != 'rl_adaptive':
        return {"status": "skipped", "message": "RL mode not enabled"}

    if correct_label not in ("REAL", "FAKE", "SUSPICIOUS"):
        raise HTTPException(status_code=400, detail="Label must be REAL, FAKE, or SUSPICIOUS")

    try:
        # Build prediction dict matching what RL system expects
        prediction = {
            'verdict': original_verdict or "Unknown",
            'video_fake_score': video_fake_score,
            'audio_fake_score': audio_fake_score,
        }

        detector.fusion_strategy.update_from_feedback(
            prediction, correct_label, user_confidence
        )
        detector.fusion_strategy.save_model()

        stats = detector.fusion_strategy.get_performance_stats()

        return {
            "status": "ok",
            "message": f"Feedback recorded: {correct_label}",
            "updated_weights": {
                "video": round(float(detector.fusion_strategy.video_weight), 3),
                "audio": round(float(detector.fusion_strategy.audio_weight), 3),
            },
            "stats": stats,
        }

    except Exception as e:
        logger.error(f"Feedback error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# ─── Run Server ──────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"\n{'='*60}")
    print(f"  DeepFake Detection Server v4.0 (WebSocket)")
    print(f"  Video model:   GenConViT-{Config.CVIT_NET}")
    print(f"  Audio model:   {Config.AUDIO_MODEL}")
    print(f"  Listening on:  http://{host}:{port}")
    print(f"  WebSocket:     ws://{host}:{port}/ws/analyze")
    print(f"  Health:        http://{host}:{port}/health")
    print(f"  Fusion mode:   {Config.FUSION_MODE}")
    print(f"{'='*60}\n")

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,  # Disable reload in production for faster startup
        workers=1,      # Single worker for GPU model sharing
    )
