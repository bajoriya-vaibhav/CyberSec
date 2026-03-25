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
import tempfile
import logging
import asyncio
from io import BytesIO
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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


# ─── Health / Status ─────────────────────────────────────────────────

@app.get("/")
async def root():
    """Server status page."""
    return {
        "status": "running",
        "service": "DeepFake Detection API",
        "version": "2.0.0",
        "models_loaded": detector is not None,
        "model_load_time_seconds": round(model_load_time, 2),
        "fusion_mode": Config.FUSION_MODE,
    }


@app.get("/health")
async def health():
    """Health check endpoint for the Android app."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ok", "models_loaded": True}


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


# ─── Prediction Endpoint ─────────────────────────────────────────────

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
    print(f"  DeepFake Detection Server")
    print(f"  Listening on: http://{host}:{port}")
    print(f"  Predict URL:  http://{host}:{port}/predict")
    print(f"  Health:       http://{host}:{port}/health")
    print(f"  Fusion mode:  {Config.FUSION_MODE}")
    print(f"{'='*60}\n")

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,  # Disable reload in production for faster startup
        workers=1,      # Single worker for GPU model sharing
    )
