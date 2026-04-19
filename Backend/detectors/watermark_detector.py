"""
LLM-based Veo Watermark Detector.

Sends a single frame to Gemini (multimodal LLM) and asks whether
a visible "Veo" watermark is present. Returns a confidence score.

Designed to run in a separate thread in parallel with GenConViT
inference, so it adds zero extra latency to the pipeline.
"""
import io
import json
import logging
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-initialized Gemini client (created once on first call)
_client = None


def _get_client():
    """Lazy-init the Gemini client using config API key."""
    global _client
    if _client is None:
        from google import genai
        from config import Config
        _client = genai.Client(api_key=Config.GEMINI_API_KEY)
        logger.info("Watermark detector: Gemini client initialized")
    return _client


def _pil_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL image to PNG bytes for the API."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def check_watermark_llm(frame: Image.Image) -> dict:
    """
    Ask Gemini whether a visible Veo watermark is present in the frame.

    Args:
        frame: A single PIL Image (typically the first or middle frame).

    Returns:
        {
            "watermark_detected": bool,
            "confidence": float,          # 0.0 - 1.0
            "watermark_type": str,        # "veo" or "none"
            "reasoning": str,
        }
    """
    try:
        from google.genai import types
        from config import Config

        client = _get_client()

        prompt = """Look at this image carefully. Does it contain a visible "Veo" watermark or any Google Veo AI video generator watermark text overlay?

The Veo watermark is typically a semi-transparent text that says "Veo" placed in one of the corners or edges of the frame.

Respond ONLY in this exact JSON format, nothing else:
{
    "has_veo_watermark": true or false,
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}"""

        response = client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png",
                                data=_pil_to_bytes(frame)
                            )
                        )
                    ]
                )
            ]
        )

        result_text = response.text.strip()

        # Extract JSON from potential markdown wrapping
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        data = json.loads(result_text)

        has_wm = bool(data.get("has_veo_watermark", False))
        confidence = float(data.get("confidence", 0.0))
        reasoning = str(data.get("reasoning", ""))

        logger.info(
            f"LLM watermark check: detected={has_wm}, "
            f"confidence={confidence:.2f}, reason={reasoning[:80]}"
        )

        return {
            "watermark_detected": has_wm,
            "confidence": confidence,
            "watermark_type": "veo" if has_wm else "none",
            "reasoning": reasoning,
        }

    except Exception as e:
        logger.warning(f"LLM watermark check failed: {e}")
        return {
            "watermark_detected": False,
            "confidence": 0.0,
            "watermark_type": "none",
            "reasoning": f"error: {e}",
        }
