"""
Gemini-based deepfake detector using the NEW Google GenAI SDK.
"""
from google import genai
from google.genai import types
from PIL import Image
import logging
from typing import List
import json
import io

from detectors.base_detector import BaseDetector, DetectionResult
from config import Config

logger = logging.getLogger(__name__)


class GeminiDetector(BaseDetector):
    """Uses Gemini API for deepfake detection with fallback support."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro"):
        super().__init__(model_name)
        self.api_key = api_key
        self.load_model()

    def load_model(self) -> None:
        try:
            logger.info(f"Initializing Gemini client: {self.model_name}")
            self.client = genai.Client(api_key=self.api_key)
            self._is_loaded = True
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")
            self._is_loaded = False
            raise

    def _pil_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL image to bytes."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def detect_image(self, image: Image.Image) -> DetectionResult:
        if not self._is_loaded:
            raise RuntimeError("Gemini model not loaded")

        prompt = """Analyze this image for signs of AI-generated or deepfake manipulation.

**Focus on:**
1. Facial inconsistencies (eyes, teeth, skin texture, asymmetry)
2. Lighting and shadow anomalies
3. Background artifacts and unrealistic elements
4. Edge artifacts around face boundaries
5. Unnatural features or distortions
6. AI generation patterns (smoothing, noise patterns)

**Be conservative:** Only flag as deepfake if you find clear evidence of manipulation.

Respond strictly in JSON format:
{
    "is_deepfake": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation with specific observations",
    "suspicious_features": ["feature1", "feature2"]
}"""

        try:
            logger.info("Calling Gemini API for image analysis...")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=prompt),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=self._pil_to_bytes(image)
                                )
                            )
                        ]
                    )
                ]
            )

            result_text = response.text.strip()
            
            # Extract JSON from potential markdown
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result_data = json.loads(result_text)

            is_fake = result_data.get("is_deepfake", False)
            confidence = float(result_data.get("confidence", 0.5))

            fake_prob = confidence if is_fake else 1.0 - confidence
            real_prob = 1.0 - fake_prob

            logger.info(f"Gemini image analysis successful: fake={fake_prob:.3f}")

            return DetectionResult(
                fake_probability=fake_prob,
                real_probability=real_prob,
                metadata={
                    "source": "gemini",
                    "model": self.model_name,
                    "reasoning": result_data.get("reasoning", "N/A"),
                    "suspicious_features": result_data.get("suspicious_features", [])
                }
            )

        except Exception as e:
            logger.error(f"Gemini image detection failed: {e}")
            raise

    def detect_video_frames(self, frames: List[Image.Image]) -> DetectionResult:
        if not self._is_loaded:
            raise RuntimeError("Gemini model not loaded")
            
        if not frames:
            raise ValueError("No frames provided")

        # Select more frames for better temporal analysis
        num_frames = len(frames)
        max_frames = Config.GEMINI_FRAMES
        
        if num_frames <= max_frames:
            selected_frames = frames
            indices = list(range(num_frames))
        else:
            # Evenly distributed frames
            indices = [int(i * num_frames / max_frames) for i in range(max_frames)]
            selected_frames = [frames[i] for i in indices]

        prompt = f"""Analyze these {len(selected_frames)} video frames (chronologically ordered) for signs of deepfake manipulation.

**Critical Analysis Points:**
1. **Temporal Consistency:** Do facial features, expressions, and movements flow naturally across frames?
2. **Audio-Video Sync Issues:** Look for signs that audio may not match video (lip sync, timing)
3. **Edge Artifacts:** Are there unnatural boundaries around the face or hair?
4. **Lighting Consistency:** Does lighting remain coherent across frames?
5. **Facial Movement Realism:** Natural blinking, micro-expressions, head movements?
6. **Background Stability:** Does the background behave naturally?
7. **Deepfake Telltale Signs:** Warping, flickering, unnatural smoothing, face-swap artifacts

**Important:** If you notice audio-video desynchronization or inconsistent facial movements across frames, this is a STRONG indicator of deepfake manipulation.

Respond strictly in JSON format:
{{
    "is_deepfake": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of temporal analysis and specific frame observations",
    "temporal_inconsistencies": ["specific issues found across frames"],
    "audio_video_sync_issues": ["any sync problems detected"]
}}"""

        try:
            logger.info(f"Calling Gemini API for video analysis ({len(selected_frames)} frames)...")
            
            # Build parts list: text prompt + all frame images
            parts = [types.Part(text=prompt)]
            for i, frame in enumerate(selected_frames):
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=self._pil_to_bytes(frame)
                        )
                    )
                )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=parts
                    )
                ]
            )

            result_text = response.text.strip()
            
            # Extract JSON from potential markdown
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result_data = json.loads(result_text)

            is_fake = result_data.get("is_deepfake", False)
            confidence = float(result_data.get("confidence", 0.5))

            fake_prob = confidence if is_fake else 1.0 - confidence
            real_prob = 1.0 - fake_prob

            logger.info(f"Gemini video analysis successful: fake={fake_prob:.3f}")

            return DetectionResult(
                fake_probability=fake_prob,
                real_probability=real_prob,
                metadata={
                    "source": "gemini",
                    "model": self.model_name,
                    "frames_analyzed": len(selected_frames),
                    "reasoning": result_data.get("reasoning", "N/A"),
                    "temporal_inconsistencies": result_data.get("temporal_inconsistencies", []),
                    "audio_video_sync_issues": result_data.get("audio_video_sync_issues", [])
                }
            )

        except Exception as e:
            logger.error(f"Gemini video detection failed: {e}")
            raise

    def detect(self, input_data) -> DetectionResult:
        if isinstance(input_data, list):
            return self.detect_video_frames(input_data)
        elif isinstance(input_data, Image.Image):
            return self.detect_image(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")