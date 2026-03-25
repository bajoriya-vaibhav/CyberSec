"""
Enhanced fusion strategies with security-first approach and mismatch detection.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

from detectors.base_detector import DetectionResult

logger = logging.getLogger(__name__)


class FusionStrategy(ABC):
    """Abstract base class for fusion strategies."""
    
    @abstractmethod
    def fuse(self, video_result: DetectionResult, 
             audio_result: Optional[DetectionResult]) -> DetectionResult:
        """
        Fuse results from multiple modalities.
        
        Args:
            video_result: Result from video analysis
            audio_result: Result from audio analysis (may be None)
            
        Returns:
            Fused DetectionResult
        """
        pass


class SecurityFirstFusionStrategy(FusionStrategy):
    """
    Security-first fusion that detects sophisticated attacks.
    
    Key features:
    1. Detects mismatch (one real, one fake) - ALERTS as SUSPICIOUS
    2. Uses conservative approach - if EITHER is fake, flag it
    3. Provides detailed threat analysis
    """
    
    def __init__(self, mismatch_threshold: float = 0.3):
        """
        Initialize security-first fusion.
        
        Args:
            mismatch_threshold: Threshold for detecting disagreement between modalities
        """
        self.mismatch_threshold = mismatch_threshold
        logger.info(f"Security-first fusion initialized (mismatch_threshold={mismatch_threshold})")
    
    def fuse(self, video_result: DetectionResult, 
             audio_result: Optional[DetectionResult]) -> DetectionResult:
        """
        Security-first fusion with mismatch detection.
        """
        # If no audio, use video only
        if audio_result is None or 'error' in audio_result.metadata:
            logger.info("No valid audio result, using video only")
            return DetectionResult(
                fake_probability=video_result.fake_probability,
                real_probability=video_result.real_probability,
                metadata={
                    'fusion_type': 'video_only',
                    'video_result': video_result.to_dict(),
                    'security_alert': None
                }
            )
        
        # Detect mismatch (sophisticated attack)
        video_is_fake = video_result.fake_probability > 0.5
        audio_is_fake = audio_result.fake_probability > 0.5
        
        # Calculate disagreement magnitude
        disagreement = abs(video_result.fake_probability - audio_result.fake_probability)
        
        security_alert = None
        threat_vector = None
        
        # CASE 1: Mismatch detected (one fake, one real)
        if video_is_fake != audio_is_fake and disagreement > self.mismatch_threshold:
            security_alert = "MISMATCH_DETECTED"
            
            if video_is_fake and not audio_is_fake:
                threat_vector = "FACE_SWAP_ATTACK"
                logger.warning("Security Alert: Possible face swap with real audio!")
            else:
                threat_vector = "VOICE_CLONE_ATTACK"
                logger.warning("Security Alert: Possible voice cloning with real video!")
            
            # Conservative: Take the higher fake probability
            fused_fake = max(video_result.fake_probability, audio_result.fake_probability)
            
        # CASE 2: Both agree it's fake
        elif video_is_fake and audio_is_fake:
            # Both fake - high confidence it's a deepfake
            # Use weighted average
            fused_fake = 0.7 * video_result.fake_probability + 0.3 * audio_result.fake_probability
            threat_vector = "FULL_DEEPFAKE"
            
        # CASE 3: Both agree it's real
        elif not video_is_fake and not audio_is_fake:
            # Both real - likely authentic
            fused_fake = 0.7 * video_result.fake_probability + 0.3 * audio_result.fake_probability
            
        # CASE 4: Close to boundary (ambiguous)
        else:
            # Use conservative max approach for safety
            fused_fake = max(video_result.fake_probability, audio_result.fake_probability)
            if disagreement > 0.2:
                security_alert = "LOW_CONFIDENCE"
        
        fused_real = 1.0 - fused_fake
        
        logger.info(f"Security fusion: fake={fused_fake:.3f}, alert={security_alert}")
        
        return DetectionResult(
            fake_probability=fused_fake,
            real_probability=fused_real,
            metadata={
                'fusion_type': 'security_first',
                'security_alert': security_alert,
                'threat_vector': threat_vector,
                'disagreement': disagreement,
                'video_result': video_result.to_dict(),
                'audio_result': audio_result.to_dict()
            }
        )


class WeightedFusionStrategy(FusionStrategy):
    """Weighted averaging of modality results."""
    
    def __init__(self, video_weight: float = 0.7, audio_weight: float = 0.3):
        """
        Initialize with weights.
        
        Args:
            video_weight: Weight for video result (0-1)
            audio_weight: Weight for audio result (0-1)
        """
        assert abs(video_weight + audio_weight - 1.0) < 0.001, \
            "Weights must sum to 1.0"
        
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        logger.info(f"Fusion weights: video={video_weight}, audio={audio_weight}")
    
    def fuse(self, video_result: DetectionResult, 
             audio_result: Optional[DetectionResult]) -> DetectionResult:
        """
        Weighted fusion of video and audio results.
        Prioritizes video since audio can be real background recording.
        """
        if audio_result is None or 'error' in audio_result.metadata:
            logger.info("No valid audio result, using video only")
            return DetectionResult(
                fake_probability=video_result.fake_probability,
                real_probability=video_result.real_probability,
                metadata={
                    'fusion_type': 'video_only',
                    'video_result': video_result.to_dict()
                }
            )
        
        # Weighted average
        fused_fake = (self.video_weight * video_result.fake_probability + 
                     self.audio_weight * audio_result.fake_probability)
        fused_real = 1.0 - fused_fake
        
        logger.info(f"Fused result: fake={fused_fake:.3f}, real={fused_real:.3f}")
        
        return DetectionResult(
            fake_probability=fused_fake,
            real_probability=fused_real,
            metadata={
                'fusion_type': 'weighted',
                'video_weight': self.video_weight,
                'audio_weight': self.audio_weight,
                'video_result': video_result.to_dict(),
                'audio_result': audio_result.to_dict()
            }
        )


class MaxFusionStrategy(FusionStrategy):
    """Take the maximum fake probability (most conservative)."""
    
    def fuse(self, video_result: DetectionResult, 
             audio_result: Optional[DetectionResult]) -> DetectionResult:
        """Take max fake probability across modalities."""
        if audio_result is None or 'error' in audio_result.metadata:
            return video_result
        
        fused_fake = max(video_result.fake_probability, 
                        audio_result.fake_probability)
        fused_real = 1.0 - fused_fake
        
        return DetectionResult(
            fake_probability=fused_fake,
            real_probability=fused_real,
            metadata={
                'fusion_type': 'max',
                'video_result': video_result.to_dict(),
                'audio_result': audio_result.to_dict()
            }
        )


class AdaptiveFusionStrategy(FusionStrategy):
    """
    Adaptive fusion that adjusts weights based on confidence.
    More confident modality gets higher weight.
    """
    
    def __init__(self, base_video_weight: float = 0.7):
        self.base_video_weight = base_video_weight
    
    def fuse(self, video_result: DetectionResult, 
             audio_result: Optional[DetectionResult]) -> DetectionResult:
        """Adaptively weight based on confidence."""
        if audio_result is None or 'error' in audio_result.metadata:
            return video_result
        
        video_conf = video_result.confidence
        audio_conf = audio_result.confidence
        
        # Adjust weights based on relative confidence
        total_conf = video_conf + audio_conf
        if total_conf > 0:
            video_weight = (video_conf / total_conf) * self.base_video_weight + (1 - self.base_video_weight) * 0.5
            audio_weight = 1.0 - video_weight
        else:
            video_weight = self.base_video_weight
            audio_weight = 1.0 - video_weight
        
        fused_fake = (video_weight * video_result.fake_probability + 
                     audio_weight * audio_result.fake_probability)
        fused_real = 1.0 - fused_fake
        
        logger.info(f"Adaptive fusion: video_weight={video_weight:.3f}, audio_weight={audio_weight:.3f}")
        
        return DetectionResult(
            fake_probability=fused_fake,
            real_probability=fused_real,
            metadata={
                'fusion_type': 'adaptive',
                'video_weight': video_weight,
                'audio_weight': audio_weight,
                'video_confidence': video_conf,
                'audio_confidence': audio_conf,
                'video_result': video_result.to_dict(),
                'audio_result': audio_result.to_dict()
            }
        )
