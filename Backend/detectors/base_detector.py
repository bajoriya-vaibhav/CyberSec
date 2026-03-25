"""
Base detector class for abstraction.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DetectionResult:
    """Structured result from a detector."""
    
    def __init__(self, fake_probability: float, real_probability: float, 
                 metadata: Dict[str, Any] = None):
        self.fake_probability = max(0.0, min(1.0, fake_probability))
        self.real_probability = max(0.0, min(1.0, real_probability))
        self.metadata = metadata or {}
    
    @property
    def is_fake(self) -> bool:
        """Returns True if classified as fake."""
        return self.fake_probability > self.real_probability
    
    @property
    def confidence(self) -> float:
        """Confidence of the prediction (0-1)."""
        return max(self.fake_probability, self.real_probability)
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'fake_probability': self.fake_probability,
            'real_probability': self.real_probability,
            'is_fake': self.is_fake,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    def __repr__(self):
        return f"DetectionResult(fake={self.fake_probability:.3f}, real={self.real_probability:.3f})"


class BaseDetector(ABC):
    """Abstract base class for all detectors."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pipeline = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the ML model."""
        pass
    
    @abstractmethod
    def detect(self, input_data: Any) -> DetectionResult:
        """
        Run detection on input data.
        
        Args:
            input_data: Data to analyze (type depends on detector)
            
        Returns:
            DetectionResult object
        """
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"
