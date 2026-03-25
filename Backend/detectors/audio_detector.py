"""
Audio-based deepfake detector.
"""
import os
from typing import Optional
from transformers import pipeline
import logging

from detectors.base_detector import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class AudioDetector(BaseDetector):
    """Detects deepfakes in audio using transformer models."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.load_model()
    
    def load_model(self) -> None:
        """Load the audio classification model."""
        try:
            logger.info(f"Loading audio model: {self.model_name}")
            self.pipeline = pipeline("audio-classification", model=self.model_name)
            self._is_loaded = True
            logger.info("Audio model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            self._is_loaded = False
            raise
    
    def detect(self, audio_path: str) -> DetectionResult:
        """
        Detect if audio is fake.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            DetectionResult with probabilities
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        if not audio_path or not os.path.exists(audio_path):
            logger.warning("Invalid audio path provided")
            return DetectionResult(
                fake_probability=0.5,
                real_probability=0.5,
                metadata={'error': 'Invalid audio path'}
            )
        
        try:
            logger.info(f"Analyzing audio: {audio_path}")
            results = self.pipeline(audio_path)
            
            # Parse results
            fake_prob = 0.0
            real_prob = 0.0
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if 'fake' in label or 'spoof' in label:
                    fake_prob = score
                elif 'real' in label or 'bonafide' in label:
                    real_prob = score
            
            # Normalize if needed
            if fake_prob == 0.0 and real_prob == 0.0:
                # If we couldn't find clear labels, take top result
                top_result = results[0]
                if 'fake' in top_result['label'].lower():
                    fake_prob = top_result['score']
                    real_prob = 1.0 - fake_prob
                else:
                    real_prob = top_result['score']
                    fake_prob = 1.0 - real_prob
            elif fake_prob > 0.0 and real_prob == 0.0:
                real_prob = 1.0 - fake_prob
            elif real_prob > 0.0 and fake_prob == 0.0:
                fake_prob = 1.0 - real_prob
            
            logger.info(f"Audio analysis: fake={fake_prob:.3f}, real={real_prob:.3f}")
            
            return DetectionResult(
                fake_probability=fake_prob,
                real_probability=real_prob,
                metadata={'raw_results': results}
            )
            
        except Exception as e:
            logger.error(f"Error in audio detection: {e}")
            return DetectionResult(
                fake_probability=0.5,
                real_probability=0.5,
                metadata={'error': str(e)}
            )
