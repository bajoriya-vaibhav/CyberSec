"""
Simple Reinforcement Learning system for adaptive deepfake detection.
Uses gradient-based learning WITHOUT external libraries.
Learns optimal fusion weights from user feedback.
"""
import json
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class FeedbackDatabase:
    """Simple file-based feedback storage."""
    
    def __init__(self, filepath: str = "feedback_data.jsonl"):
        self.filepath = filepath
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create feedback file if it doesn't exist."""
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                pass
    
    def add_feedback(self, prediction: Dict[str, Any], user_label: str, 
                    confidence: float, metadata: Dict[str, Any] = None):
        """
        Store user feedback.
        
        Args:
            prediction: The model's prediction
            user_label: User's correction (REAL/FAKE/SUSPICIOUS)
            confidence: How confident the user is (0-1)
            metadata: Additional context
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': convert_to_serializable(prediction),
            'user_label': user_label,
            'user_confidence': float(confidence),
            'metadata': convert_to_serializable(metadata) if metadata else {}
        }
        
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')
        
        logger.info(f"Feedback stored: {user_label} (confidence: {confidence})")
    
    def get_all_feedback(self):
        """Retrieve all feedback entries."""
        feedback = []
        with open(self.filepath, 'r') as f:
            for line in f:
                if line.strip():
                    feedback.append(json.loads(line))
        return feedback
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        feedback = self.get_all_feedback()
        if not feedback:
            return {'total': 0, 'accuracy': 0.0, 'correct': 0, 'incorrect': 0}
        
        total = len(feedback)
        correct = sum(1 for f in feedback 
                     if f['prediction']['verdict'] == f['user_label'])
        
        return {
            'total': total,
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'incorrect': total - correct
        }


class RLAdaptiveFusion:
    """
    Simple Reinforcement Learning based adaptive fusion.
    Learns optimal weights from user feedback using gradient descent.
    NO external libraries - pure Python implementation!
    """
    
    def __init__(self, initial_video_weight: float = 0.8, 
                 learning_rate: float = 0.05,
                 exploration_rate: float = 0.0):
        """
        Initialize RL-based fusion.
        
        Args:
            initial_video_weight: Starting weight for video (default 0.8)
            learning_rate: How fast to adapt (0.05 = 5% adjustment)
            exploration_rate: Probability of trying random weights (not used in simple version)
        """
        self.video_weight = initial_video_weight
        self.audio_weight = 1.0 - initial_video_weight
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.feedback_db = FeedbackDatabase()
        
        # Performance tracking
        self.rewards = []
        self.weight_history = []
        
        logger.info(f"Simple RL Fusion initialized: video={self.video_weight:.2f}, "
                   f"audio={self.audio_weight:.2f}, lr={self.learning_rate}")
    
    def fuse(self, video_result, audio_result):
        """
        Fuse results using current learned weights.
        
        Args:
            video_result: DetectionResult from video
            audio_result: DetectionResult from audio (optional)
            
        Returns:
            DetectionResult
        """
        from detectors.base_detector import DetectionResult
        
        if audio_result is None or 'error' in audio_result.metadata:
            logger.info("No valid audio, using video only")
            return DetectionResult(
                fake_probability=video_result.fake_probability,
                real_probability=video_result.real_probability,
                metadata={
                    'fusion_type': 'rl_adaptive',
                    'video_only': True,
                    'video_result': video_result.to_dict()
                }
            )
        
        # Use learned weights
        fused_fake = (self.video_weight * video_result.fake_probability + 
                     self.audio_weight * audio_result.fake_probability)
        fused_real = 1.0 - fused_fake
        
        logger.info(f"RL Fusion: video_w={self.video_weight:.3f}, "
                   f"audio_w={self.audio_weight:.3f}, fake={fused_fake:.3f}")
        
        return DetectionResult(
            fake_probability=fused_fake,
            real_probability=fused_real,
            metadata={
                'fusion_type': 'rl_adaptive',
                'video_weight': self.video_weight,
                'audio_weight': self.audio_weight,
                'video_result': video_result.to_dict(),
                'audio_result': audio_result.to_dict()
            }
        )
    
    def update_from_feedback(self, prediction: Dict[str, Any], 
                            user_label: str, user_confidence: float = 1.0):
        """
        Update weights based on user feedback using simple gradient descent.
        
        Args:
            prediction: The prediction that was made
            user_label: User's correction (REAL/FAKE/SUSPICIOUS)
            user_confidence: How confident the user is (0-1)
        """
        # Store feedback
        self.feedback_db.add_feedback(prediction, user_label, user_confidence)
        
        # Calculate reward
        was_correct = (prediction['verdict'] == user_label)
        
        # Reward: +1 for correct, -1 for incorrect (weighted by user confidence)
        reward = (1.0 if was_correct else -1.0) * user_confidence
        self.rewards.append(reward)
        
        logger.info(f"Feedback received: {user_label}, reward={reward:.2f}")
        
        # Update weights using gradient-based RL
        if not was_correct and 'video_fake_score' in prediction and 'audio_fake_score' in prediction:
            video_score = prediction['video_fake_score']
            audio_score = prediction['audio_fake_score']
            
            if audio_score is not None:
                # Determine which modality was more accurate
                user_is_fake = (user_label == 'FAKE' or user_label == 'SUSPICIOUS')
                user_fake_prob = 0.75 if user_is_fake else 0.25
                
                video_error = abs(video_score - user_fake_prob)
                audio_error = abs(audio_score - user_fake_prob)
                
                # Adjust weights: increase weight of more accurate modality
                if video_error < audio_error:
                    # Video was more accurate, increase its weight
                    adjustment = self.learning_rate * user_confidence
                    self.video_weight = np.clip(self.video_weight + adjustment, 0.5, 0.95)
                    logger.info(f"Video was more accurate, increased weight by {adjustment:.3f}")
                else:
                    # Audio was more accurate, decrease video weight (increase audio)
                    adjustment = self.learning_rate * user_confidence
                    self.video_weight = np.clip(self.video_weight - adjustment, 0.5, 0.95)
                    logger.info(f"Audio was more accurate, decreased video weight by {adjustment:.3f}")
                
                self.audio_weight = 1.0 - self.video_weight
                
                self.weight_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'video_weight': self.video_weight,
                    'audio_weight': self.audio_weight,
                    'reward': reward
                })
                
                logger.info(f"Weights updated: video={self.video_weight:.3f}, "
                           f"audio={self.audio_weight:.3f}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.feedback_db.get_stats()
        
        if self.rewards:
            stats['avg_reward'] = np.mean(self.rewards[-100:])  # Last 100
            stats['recent_accuracy'] = (np.mean([r > 0 for r in self.rewards[-20:]]) 
                                       if len(self.rewards) >= 20 else None)
        
        stats['current_weights'] = {
            'video': self.video_weight,
            'audio': self.audio_weight
        }
        
        return stats
    
    def save_model(self, filepath: str = "rl_weights.json"):
        """Save learned weights."""
        model_data = {
            'video_weight': self.video_weight,
            'audio_weight': self.audio_weight,
            'performance': self.get_performance_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"RL weights saved to {filepath}")
    
    def load_model(self, filepath: str = "rl_weights.json"):
        """Load previously learned weights."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.video_weight = model_data['video_weight']
            self.audio_weight = model_data['audio_weight']
            
            logger.info(f"RL weights loaded: video={self.video_weight:.3f}, "
                       f"audio={self.audio_weight:.3f}")
            return True
        logger.info("No saved weights found, using initial weights")
        return False
