"""
GenConViT (CVIT) deepfake detector with person/face detection.

Detection hierarchy for each frame:
  1. dlib face detection (best for GenConViT – trained on face crops)
  2. OpenCV Haar cascade face detection (fast fallback)
  3. OpenCV upper body detection (for when face isn't visible)
  4. OpenCV HOG person detection (full body)
  5. None → no person detected → frame skipped

Also provides has_person() for lightweight presence checking.
"""
import os
import logging
import numpy as np
import cv2
import torch
from typing import List, Optional, Tuple
from PIL import Image
from torchvision import transforms

from detectors.base_detector import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_BACKEND_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"


class CvitDetector(BaseDetector):
    """
    Deepfake detector using GenConViT (ED + VAE ensemble).
    
    Class mapping (from training): 0 = FAKE, 1 = REAL
    """
    
    def __init__(
        self,
        net: str = "genconvit",
        ed_weight: str = "genconvit_ed_inference",
        vae_weight: str = "genconvit_vae_inference",
        weight_dir: str = None,
        fp16: bool = False,
    ):
        super().__init__(model_name=f"GenConViT-{net}")
        self.net = net
        self.ed_weight = ed_weight
        self.vae_weight = vae_weight
        self.weight_dir = weight_dir or os.path.join(_BACKEND_DIR, "weight")
        self.fp16 = fp16
        self.model = None
        self.face_model = None
        
        # ImageNet normalization
        self.normalize = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # OpenCV detectors (initialized in load_model)
        self.haar_face = None
        self.haar_upper_body = None
        self.hog_person = None
        
        self.load_model()
    
    def load_model(self) -> None:
        """Load GenConViT model, face_recognition, and OpenCV detectors."""
        try:
            import yaml
            import dlib
            import face_recognition as fr
            
            self.face_recognition = fr
            self.face_model = "cnn" if dlib.DLIB_USE_CUDA else "hog"
            logger.info(f"dlib face backend: {self.face_model}")
            
            # OpenCV cascade classifiers (fast, no extra files needed)
            self.haar_face = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.haar_upper_body = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            )
            # HOG person detector
            self.hog_person = cv2.HOGDescriptor()
            self.hog_person.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            logger.info("OpenCV Haar + HOG person detectors loaded")
            
            # Load GenConViT config
            config_path = os.path.join(_BACKEND_DIR, "model", "config.yaml")
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loading GenConViT ({self.net}) from {self.weight_dir}")
            
            from model.genconvit import GenConViT
            self.model = GenConViT(
                config,
                ed=self.ed_weight,
                vae=self.vae_weight,
                net=self.net,
                fp16=self.fp16,
                weight_dir=self.weight_dir,
            )
            self.model.to(device)
            self.model.eval()
            if self.fp16:
                self.model.half()
            
            self._is_loaded = True
            logger.info(f"GenConViT loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load GenConViT: {e}", exc_info=True)
            self._is_loaded = False
            raise
    
    # ─── Lightweight Person Presence Check ───────────────────────
    
    def has_person(self, image: Image.Image) -> dict:
        """
        Fast person/face presence check. No model inference.
        Uses only OpenCV Haar cascades (milliseconds).
        
        Returns:
            {"person_detected": bool, "type": str, "count": int}
        """
        frame_rgb = np.array(image)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Fast Haar face check
        if self.haar_face is not None:
            faces = self.haar_face.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            if len(faces) > 0:
                logger.debug(f"has_person: {len(faces)} face(s) via Haar")
                return {"person_detected": True, "type": "face", "count": len(faces)}
        
        # 2. Upper body check
        if self.haar_upper_body is not None:
            # Resize for speed
            h, w = gray.shape
            scale = min(1.0, 480.0 / max(h, w))
            small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
            bodies = self.haar_upper_body.detectMultiScale(
                small, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
            )
            if len(bodies) > 0:
                logger.debug(f"has_person: {len(bodies)} body(ies) via Haar upper body")
                return {"person_detected": True, "type": "upper_body", "count": len(bodies)}
        
        # 3. HOG full person check
        if self.hog_person is not None:
            h, w = frame_rgb.shape[:2]
            scale = min(1.0, 400.0 / max(h, w))
            small = cv2.resize(frame_rgb, None, fx=scale, fy=scale) if scale < 1.0 else frame_rgb
            rects, _ = self.hog_person.detectMultiScale(
                small, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            if len(rects) > 0:
                logger.debug(f"has_person: {len(rects)} person(s) via HOG")
                return {"person_detected": True, "type": "person", "count": len(rects)}
        
        return {"person_detected": False, "type": "none", "count": 0}
    
    # ─── Region Extraction (for model input) ────────────────────
    
    def _extract_region(self, image: Image.Image) -> Tuple[Optional[np.ndarray], str]:
        """
        Extract the best 224x224 RGB region for GenConViT analysis.
        
        Hierarchy:
          1. dlib face detection (best for model accuracy)
          2. OpenCV Haar face detection (faster fallback)
          3. OpenCV upper body detection
          4. OpenCV HOG person detection (crop upper half)
          5. None → no person detected
        
        Returns:
            (224x224_rgb_array_or_None, detection_method)
        """
        frame_rgb = np.array(image)
        
        # ── Level 1: dlib face detection ──
        crop = self._try_dlib_face(frame_rgb, upsample=0)
        if crop is not None:
            return crop, "face_dlib"
        
        crop = self._try_dlib_face(frame_rgb, upsample=1)
        if crop is not None:
            return crop, "face_dlib_upsample"
        
        # ── Level 2: OpenCV Haar face detection ──
        crop = self._try_haar_face(frame_rgb)
        if crop is not None:
            return crop, "face_haar"
        
        # ── Level 3: Upper body detection ──
        crop = self._try_upper_body(frame_rgb)
        if crop is not None:
            return crop, "upper_body"
        
        # ── Level 4: HOG person detection ──
        crop = self._try_hog_person(frame_rgb)
        if crop is not None:
            return crop, "person_hog"
        
        # ── No person detected ──
        return None, "no_person"
    
    def _try_dlib_face(self, frame_rgb: np.ndarray, upsample: int = 0) -> Optional[np.ndarray]:
        """Detect face using face_recognition (dlib). Returns 224x224 RGB crop."""
        try:
            face_locations = self.face_recognition.face_locations(
                frame_rgb, number_of_times_to_upsample=upsample, model=self.face_model
            )
            if not face_locations:
                return None
            
            best = max(face_locations, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
            top, right, bottom, left = best
            if (bottom - top) < 20 or (right - left) < 20:
                return None
            
            crop = frame_rgb[top:bottom, left:right]
            return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.debug(f"dlib face error: {e}")
            return None
    
    def _try_haar_face(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Detect face using OpenCV Haar cascade. Returns 224x224 RGB crop."""
        if self.haar_face is None:
            return None
        try:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = self.haar_face.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) == 0:
                return None
            
            # Largest face
            best = max(faces, key=lambda r: r[2] * r[3])
            x, y, w, h = best
            if w < 20 or h < 20:
                return None
            
            crop = frame_rgb[y:y+h, x:x+w]
            logger.debug(f"Haar face: {w}x{h}px at ({x},{y})")
            return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.debug(f"Haar face error: {e}")
            return None
    
    def _try_upper_body(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Detect upper body using Haar cascade. Returns 224x224 RGB crop."""
        if self.haar_upper_body is None:
            return None
        try:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            h_img, w_img = gray.shape
            scale = min(1.0, 480.0 / max(h_img, w_img))
            small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
            
            bodies = self.haar_upper_body.detectMultiScale(
                small, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
            )
            if len(bodies) == 0:
                return None
            
            best = max(bodies, key=lambda r: r[2] * r[3])
            x, y, w, h = best
            # Scale back to original coords
            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            
            # Clamp to image bounds
            x, y = max(0, x), max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            if w < 30 or h < 30:
                return None
            
            crop = frame_rgb[y:y+h, x:x+w]
            logger.debug(f"Upper body: {w}x{h}px at ({x},{y})")
            return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.debug(f"Upper body error: {e}")
            return None
    
    def _try_hog_person(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Detect full person using HOG. Crops upper half. Returns 224x224 RGB."""
        if self.hog_person is None:
            return None
        try:
            h_img, w_img = frame_rgb.shape[:2]
            scale = min(1.0, 400.0 / max(h_img, w_img))
            small = cv2.resize(frame_rgb, None, fx=scale, fy=scale) if scale < 1.0 else frame_rgb
            
            rects, _ = self.hog_person.detectMultiScale(
                small, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            if len(rects) == 0:
                return None
            
            best = max(rects, key=lambda r: r[2] * r[3])
            x, y, w, h = best
            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            
            # Focus on upper half (head/face area)
            upper_h = max(h // 2, 50)
            x, y = max(0, x), max(0, y)
            w = min(w, w_img - x)
            upper_h = min(upper_h, h_img - y)
            
            if w < 30 or upper_h < 30:
                return None
            
            crop = frame_rgb[y:y+upper_h, x:x+w]
            logger.debug(f"HOG person: upper body {w}x{upper_h}px at ({x},{y})")
            return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.debug(f"HOG person error: {e}")
            return None
    
    # ─── Preprocessing ──────────────────────────────────────────
    
    def _preprocess_faces(self, faces: np.ndarray) -> torch.Tensor:
        """Convert (N, 224, 224, 3) uint8 array to normalized tensor."""
        df_tensor = torch.tensor(faces, device=device).float()
        df_tensor = df_tensor.permute((0, 3, 1, 2))  # NHWC → NCHW
        for i in range(len(df_tensor)):
            df_tensor[i] = self.normalize(df_tensor[i] / 255.0)
        if self.fp16:
            df_tensor = df_tensor.half()
        return df_tensor
    
    # ─── Prediction ──────────────────────────────────────────────
    
    def _predict_batch(self, face_tensor: torch.Tensor) -> Tuple[float, float, dict]:
        """Run GenConViT on batch of preprocessed tensors."""
        with torch.no_grad():
            raw_output = self.model(face_tensor)
            y_pred = torch.sigmoid(raw_output.squeeze())
            
            if y_pred.dim() == 1:
                y_pred = y_pred.unsqueeze(0)
            
            mean_val = torch.mean(y_pred, dim=0)
            idx = torch.argmax(mean_val).item()
            val = torch.max(mean_val).item()
            raw_fake = mean_val[0].item()
            raw_real = mean_val[1].item()
            
            if idx == 1:
                real_prob = val
                fake_prob = 1.0 - val
            else:
                fake_prob = val
                real_prob = 1.0 - val
            
            logger.info(
                f"GenConViT: sigmoid=[{raw_fake:.4f}, {raw_real:.4f}], "
                f"{'REAL' if idx == 1 else 'FAKE'}, "
                f"fake={fake_prob:.4f}, real={real_prob:.4f}"
            )
        
        return fake_prob, real_prob, {
            'source': 'cvit',
            'model': self.net,
            'num_inputs': face_tensor.shape[0],
            'raw_class_idx': idx,
            'raw_sigmoid': [raw_fake, raw_real],
            'raw_max_val': val,
        }
    
    # ─── Public API ──────────────────────────────────────────────
    
    def detect_single_frame(self, frame: Image.Image) -> DetectionResult:
        """Analyze a single frame. Returns no_person if nobody found."""
        if not self._is_loaded:
            raise RuntimeError("GenConViT model not loaded")
        
        region, method = self._extract_region(frame)
        
        if region is None:
            logger.debug("No person detected in frame")
            return DetectionResult(
                fake_probability=0.5,
                real_probability=0.5,
                metadata={
                    'source': 'cvit',
                    'person_detected': False,
                    'extraction_method': method,
                }
            )
        
        logger.debug(f"Frame region via: {method}")
        faces = np.expand_dims(region, axis=0)
        tensor = self._preprocess_faces(faces)
        fake_prob, real_prob, meta = self._predict_batch(tensor)
        meta['person_detected'] = True
        meta['extraction_method'] = method
        
        return DetectionResult(
            fake_probability=fake_prob,
            real_probability=real_prob,
            metadata=meta
        )
    
    def detect(self, frames: List[Image.Image]) -> DetectionResult:
        """Analyze multiple frames. Skips frames with no person."""
        if not self._is_loaded:
            raise RuntimeError("GenConViT model not loaded")
        
        if not frames:
            return DetectionResult(
                fake_probability=0.5,
                real_probability=0.5,
                metadata={'error': 'No frames provided', 'person_detected': False}
            )
        
        logger.info(f"CVIT: Analyzing {len(frames)} frames")
        
        regions = []
        methods = []
        no_person_count = 0
        
        for i, frame in enumerate(frames):
            region, method = self._extract_region(frame)
            methods.append(method)
            if region is not None:
                regions.append(region)
            else:
                no_person_count += 1
                logger.debug(f"Frame {i}: no person")
        
        person_detected = len(regions) > 0
        
        if not person_detected:
            logger.warning(f"No person detected in any of {len(frames)} frames")
            return DetectionResult(
                fake_probability=0.5,
                real_probability=0.5,
                metadata={
                    'source': 'cvit',
                    'person_detected': False,
                    'frames_submitted': len(frames),
                    'no_person_frames': no_person_count,
                    'extraction_methods': methods,
                }
            )
        
        face_array = np.stack(regions, axis=0)
        tensor = self._preprocess_faces(face_array)
        fake_prob, real_prob, meta = self._predict_batch(tensor)
        
        meta['person_detected'] = True
        meta['frames_submitted'] = len(frames)
        meta['regions_found'] = len(regions)
        meta['no_person_frames'] = no_person_count
        meta['extraction_methods'] = methods
        
        method_counts = {}
        for m in methods:
            method_counts[m] = method_counts.get(m, 0) + 1
        meta['method_summary'] = method_counts
        
        return DetectionResult(
            fake_probability=fake_prob,
            real_probability=real_prob,
            metadata=meta
        )
