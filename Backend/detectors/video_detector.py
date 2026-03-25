"""
Video/Image-based deepfake detector.
"""
from typing import List
from PIL import Image
from transformers import pipeline
import logging
import numpy as np

from detectors.base_detector import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class VideoDetector(BaseDetector):
    """Detects deepfakes in video frames using vision models."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.load_model()
    
    def load_model(self) -> None:
        """Load the image classification model."""
        try:
            logger.info(f"Loading video model: {self.model_name}")
            self.pipeline = pipeline("image-classification", model=self.model_name)
            self._is_loaded = True
            logger.info("Video model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load video model: {e}")
            self._is_loaded = False
            raise
    
    def detect_single_frame(self, frame: Image.Image) -> DetectionResult:
        """
        Analyze a single frame.
        
        Args:
            frame: PIL Image
            
        Returns:
            DetectionResult for this frame
        """
        try:
            results = self.pipeline(frame)
            
            fake_prob = 0.0
            real_prob = 0.0
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if 'fake' in label or 'deepfake' in label:
                    fake_prob = score
                elif 'real' in label or 'realism' in label:
                    real_prob = score
            
            # Normalize
            if fake_prob == 0.0 and real_prob == 0.0:
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
            
            return DetectionResult(
                fake_probability=fake_prob,
                real_probability=real_prob,
                metadata={'raw_results': results}
            )
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return DetectionResult(
                fake_probability=0.5,
                real_probability=0.5,
                metadata={'error': str(e)}
            )
    
    def detect(self, frames: List[Image.Image]) -> DetectionResult:
        """
        Analyze multiple frames and aggregate results.
        
        Args:
            frames: List of PIL Images
            
        Returns:
            Aggregated DetectionResult
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        if not frames:
            logger.warning("No frames provided")
            return DetectionResult(
                fake_probability=0.5,
                real_probability=0.5,
                metadata={'error': 'No frames provided'}
            )
        
        logger.info(f"Analyzing {len(frames)} frames")
        
        frame_results = []
        for i, frame in enumerate(frames):
            result = self.detect_single_frame(frame)
            frame_results.append(result)
            logger.debug(f"Frame {i+1}/{len(frames)}: {result}")
        
        # Aggregate using multiple strategies
        fake_probs = [r.fake_probability for r in frame_results]
        real_probs = [r.real_probability for r in frame_results]
        
        # Strategy 1: Mean
        mean_fake = np.mean(fake_probs)
        mean_real = np.mean(real_probs)
        
        # Strategy 2: Max (most suspicious frame)
        max_fake = np.max(fake_probs)
        
        # Strategy 3: Weighted mean (give more weight to suspicious frames)
        weights = np.array(fake_probs)  # Frames with higher fake prob get more weight
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
        weighted_fake = np.average(fake_probs, weights=weights)
        
        # Use weighted average as primary, with max as secondary indicator
        final_fake = (weighted_fake * 0.7) + (max_fake * 0.3)
        final_real = 1.0 - final_fake
        
        logger.info(f"Video analysis: fake={final_fake:.3f}, real={final_real:.3f}")
        
        return DetectionResult(
            fake_probability=final_fake,
            real_probability=final_real,
            metadata={
                'num_frames': len(frames),
                'mean_fake': mean_fake,
                'max_fake': max_fake,
                'weighted_fake': weighted_fake,
                'frame_results': [r.to_dict() for r in frame_results]
            }
        )
