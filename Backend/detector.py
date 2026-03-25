"""
Main orchestrator for multimodal deepfake detection with Gemini integration.
"""
import logging
from typing import Dict, Any, Optional, List
from PIL import Image

from config import Config
from utils.video_processor import VideoProcessor
from utils.audio_processor import AudioProcessor
from detectors.audio_detector import AudioDetector
from detectors.video_detector import VideoDetector
from detectors.base_detector import DetectionResult
from fusion.fusion_strategy import (
    SecurityFirstFusionStrategy, 
    WeightedFusionStrategy, 
    MaxFusionStrategy,
    AdaptiveFusionStrategy,
    FusionStrategy
)
from rl_system import RLAdaptiveFusion

# Try to import Gemini detector
try:
    from detectors.gemini_detector import GeminiDetector
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Gemini detector not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO if Config.VERBOSE else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepFakeDetector:
    """
    Main detector orchestrating video and audio analysis.
    Supports both Gemini API and local models with automatic fallback.
    """
    
    def __init__(self, fusion_strategy: FusionStrategy = None):
        """
        Initialize the multimodal detector.
        
        Args:
            fusion_strategy: Strategy for combining results (default: from config)
        """
        logger.info("Initializing DeepFake Detector")
        Config.validate()
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        
        # Initialize Gemini detector if available and enabled
        self.gemini_detector = None
        if Config.USE_GEMINI and GEMINI_AVAILABLE:
            try:
                logger.info("Initializing Gemini detector...")
                self.gemini_detector = GeminiDetector(
                    api_key=Config.GEMINI_API_KEY,
                    model_name=Config.GEMINI_MODEL
                )
                logger.info("Gemini detector initialized - will be used as primary")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}. Will use local models.")
                self.gemini_detector = None
        
        # Initialize local detectors (always available as fallback)
        logger.info("Loading local models...")
        self.audio_detector = AudioDetector(Config.AUDIO_MODEL)
        self.video_detector = VideoDetector(Config.VIDEO_MODEL)
        logger.info("Local models loaded successfully")
        
        # Initialize fusion strategy based on config
        if fusion_strategy is None:
            if Config.FUSION_MODE == 'security_first':
                self.fusion_strategy = SecurityFirstFusionStrategy(
                    mismatch_threshold=Config.MISMATCH_THRESHOLD
                )
            elif Config.FUSION_MODE == 'weighted':
                self.fusion_strategy = WeightedFusionStrategy(
                    video_weight=Config.VIDEO_WEIGHT,
                    audio_weight=Config.AUDIO_WEIGHT
                )
            elif Config.FUSION_MODE == 'max':
                self.fusion_strategy = MaxFusionStrategy()
            elif Config.FUSION_MODE == 'adaptive':
                self.fusion_strategy = AdaptiveFusionStrategy()
            elif Config.FUSION_MODE == 'rl_adaptive':
                # Simple Reinforcement Learning based adaptive fusion
                self.fusion_strategy = RLAdaptiveFusion(
                    initial_video_weight=Config.VIDEO_WEIGHT,
                    learning_rate=Config.RL_LEARNING_RATE
                )
                # Load previously learned weights if enabled
                if Config.RL_LOAD_PREVIOUS:
                    self.fusion_strategy.load_model()
                logger.info("RL Adaptive Fusion initialized")
            else:
                raise ValueError(f"Unknown fusion mode: {Config.FUSION_MODE}")
        else:
            self.fusion_strategy = fusion_strategy
        
        logger.info(f"Using fusion strategy: {self.fusion_strategy.__class__.__name__}")
    
    def _analyze_video_with_gemini(self, frames: List[Image.Image]) -> Optional[DetectionResult]:
        """Try to analyze video using Gemini API."""
        if not self.gemini_detector:
            return None
        
        try:
            logger.info("Attempting Gemini video analysis...")
            result = self.gemini_detector.detect_video_frames(frames)
            logger.info("Gemini analysis successful")
            return result
        except Exception as e:
            logger.warning(f"Gemini analysis failed: {e}. Falling back to local models.")
            return None
    
    def _analyze_image_with_gemini(self, image: Image.Image) -> Optional[DetectionResult]:
        """Try to analyze image using Gemini API."""
        if not self.gemini_detector:
            return None
        
        try:
            logger.info("Attempting Gemini image analysis...")
            result = self.gemini_detector.detect_image(image)
            logger.info("Gemini analysis successful")
            return result
        except Exception as e:
            logger.warning(f"Gemini analysis failed: {e}. Falling back to local model.")
            return None
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze a video file for deepfake detection.
        Uses Gemini API if available, falls back to local models.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with detailed analysis results
        """
        logger.info(f"Starting analysis of: {video_path}")
        
        try:
            # Get video info
            video_info = self.video_processor.get_video_info(video_path)
            logger.info(f"Video info: {video_info}")
            
            # Extract frames
            frames = self.video_processor.extract_frames(
                video_path, 
                num_frames=Config.NUM_FRAMES_TO_EXTRACT,
                strategy='uniform'
            )
            
            if len(frames) < Config.MIN_FRAMES_REQUIRED:
                logger.error(f"Insufficient frames extracted: {len(frames)}")
                return self._error_response("Failed to extract sufficient frames from video")
            
            # Extract audio
            audio_path = self.audio_processor.extract_audio(
                video_path, 
                Config.TEMP_AUDIO_PATH
            )
            
            # Try Gemini first for video analysis
            video_result = self._analyze_video_with_gemini(frames)
            
            # Fallback to local model if Gemini failed
            if video_result is None:
                logger.info("Running local video analysis...")
                video_result = self.video_detector.detect(frames)
            
            # Audio analysis (always use local model - Gemini doesn't support audio yet)
            audio_result = None
            if audio_path:
                logger.info("Running audio analysis...")
                audio_result = self.audio_detector.detect(audio_path)
                # Cleanup
                self.audio_processor.cleanup_audio(audio_path)
            else:
                logger.warning("No audio available for analysis")
            
            # Fuse results
            logger.info("Fusing results...")
            fused_result = self.fusion_strategy.fuse(video_result, audio_result)
            
            # Generate response
            response = self._format_response(
                fused_result, 
                video_result, 
                audio_result,
                video_info
            )
            
            logger.info(f"Analysis complete: {response['verdict']}")
            return response
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            return self._error_response(f"Analysis failed: {str(e)}")
    
    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze a single image for deepfake detection.
        Uses Gemini API if available, falls back to local model.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting image analysis")
        
        try:
            # Try Gemini first
            result = self._analyze_image_with_gemini(image)
            
            # Fallback to local model if Gemini failed
            if result is None:
                logger.info("Running local image analysis...")
                result = self.video_detector.detect_single_frame(image)
            
            verdict = "FAKE" if result.is_fake else "REAL"
            confidence_level = self._get_confidence_level(result.confidence)
            
            return {
                'verdict': verdict,
                'confidence': float(result.confidence),
                'confidence_level': confidence_level,
                'fake_probability': float(result.fake_probability),
                'real_probability': float(result.real_probability),
                'mode': 'image_only',
                'analysis_source': result.metadata.get('source', 'local')
            }
            
        except Exception as e:
            logger.error(f"Error during image analysis: {e}", exc_info=True)
            return self._error_response(f"Image analysis failed: {str(e)}")
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze an audio file for deepfake detection.
        Always uses local model (Gemini doesn't support audio analysis yet).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting audio analysis: {audio_path}")
        
        try:
            result = self.audio_detector.detect(audio_path)
            
            verdict = "FAKE" if result.is_fake else "REAL"
            confidence_level = self._get_confidence_level(result.confidence)
            
            return {
                'verdict': verdict,
                'confidence': float(result.confidence),
                'confidence_level': confidence_level,
                'fake_probability': float(result.fake_probability),
                'real_probability': float(result.real_probability),
                'mode': 'audio_only'
            }
            
        except Exception as e:
            logger.error(f"Error during audio analysis: {e}", exc_info=True)
            return self._error_response(f"Audio analysis failed: {str(e)}")
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Determine confidence level."""
        if confidence >= Config.HIGH_CONFIDENCE_THRESHOLD:
            return "HIGH"
        elif confidence >= 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _format_response(self, fused_result, video_result, audio_result, video_info) -> Dict[str, Any]:
        """Format the final response with security alerts."""
        verdict = "FAKE" if fused_result.is_fake else "REAL"
        confidence_level = self._get_confidence_level(fused_result.confidence)
        
        # Extract security information
        security_alert = fused_result.metadata.get('security_alert')
        threat_vector = fused_result.metadata.get('threat_vector')
        
        # Override verdict if security alert detected
        if security_alert == 'MISMATCH_DETECTED':
            verdict = "SUSPICIOUS"
        
        # Determine analysis source
        analysis_source = video_result.metadata.get('source', 'local')
        
        return {
            'verdict': verdict,
            'confidence': float(fused_result.confidence),
            'confidence_level': confidence_level,
            'fake_probability': float(fused_result.fake_probability),
            'real_probability': float(fused_result.real_probability),
            'video_fake_score': float(video_result.fake_probability),
            'audio_fake_score': float(audio_result.fake_probability) if audio_result else None,
            'security_alert': security_alert,
            'threat_vector': threat_vector,
            'analysis_source': analysis_source,
            'video_info': video_info,
            'mode': 'video',
            'details': {
                'fusion_metadata': fused_result.metadata,
                'video_metadata': video_result.metadata,
                'audio_metadata': audio_result.metadata if audio_result else None
            }
        }
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response."""
        return {
            'verdict': 'ERROR',
            'confidence': 0.0,
            'confidence_level': 'NONE',
            'fake_probability': 0.5,
            'real_probability': 0.5,
            'video_fake_score': 0.5,
            'audio_fake_score': None,
            'security_alert': None,
            'threat_vector': None,
            'analysis_source': 'none',
            'error': error_message,
            'details': {}
        }
