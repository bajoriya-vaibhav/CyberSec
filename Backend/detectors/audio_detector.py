"""
Audio-based deepfake detector using Wav2Vec2.
Uses garystafford/wav2vec2-deepfake-voice-detector for detecting
AI-generated speech vs authentic human speech.

Class 0: Real (human) audio
Class 1: Fake (AI-generated) audio
"""
import os
import logging
import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from detectors.base_detector import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


class AudioDetector(BaseDetector):
    """Detects deepfake audio using fine-tuned Wav2Vec2 model."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.audio_model = None
        self.feature_extractor = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the Wav2Vec2 audio classification model."""
        try:
            logger.info(f"Loading audio model: {self.model_name}")
            
            self.audio_model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            
            self.audio_model.to(device)
            self.audio_model.eval()
            
            self._is_loaded = True
            logger.info(f"Audio model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}", exc_info=True)
            self._is_loaded = False
            raise
    
    def detect(self, audio_path: str) -> DetectionResult:
        """
        Detect if audio is AI-generated (deepfake).
        
        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
            
        Returns:
            DetectionResult with fake/real probabilities
        """
        if not self._is_loaded:
            raise RuntimeError("Audio model not loaded")
        
        if not audio_path or not os.path.exists(audio_path):
            logger.warning(f"Invalid audio path: {audio_path}")
            return DetectionResult(
                fake_probability=0.5,
                real_probability=0.5,
                metadata={'error': 'Invalid audio path'}
            )
        
        try:
            logger.info(f"Analyzing audio: {audio_path}")
            
            # Load audio at 16kHz mono (model requirement)
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Validate audio length
            duration = len(audio) / sr
            if duration < 0.5:
                logger.warning(f"Audio too short: {duration:.2f}s")
                return DetectionResult(
                    fake_probability=0.5,
                    real_probability=0.5,
                    metadata={'error': f'Audio too short ({duration:.2f}s)', 'duration': duration}
                )
            
            # Optimal range is 2.5-13s, log if outside
            if duration > 15:
                logger.info(f"Audio is {duration:.1f}s, trimming to first 13s for optimal results")
                audio = audio[:int(13 * sr)]
            
            # Preprocess for Wav2Vec2
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.audio_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Class 0 = Real, Class 1 = Fake
            prob_real = probs[0][0].item()
            prob_fake = probs[0][1].item()
            
            prediction = "fake" if prob_fake > 0.5 else "real"
            
            logger.info(
                f"Audio analysis: {prediction} "
                f"(real={prob_real:.4f}, fake={prob_fake:.4f}, "
                f"duration={duration:.1f}s)"
            )
            
            return DetectionResult(
                fake_probability=prob_fake,
                real_probability=prob_real,
                metadata={
                    'source': 'wav2vec2',
                    'prediction': prediction,
                    'duration_seconds': round(duration, 2),
                    'raw_probs': {'real': round(prob_real, 4), 'fake': round(prob_fake, 4)},
                }
            )
            
        except Exception as e:
            logger.error(f"Error in audio detection: {e}", exc_info=True)
            return DetectionResult(
                fake_probability=0.5,
                real_probability=0.5,
                metadata={'error': str(e)}
            )
