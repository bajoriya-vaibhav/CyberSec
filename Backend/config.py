"""
Configuration module for deepfake detection system.
"""

class Config:
    """Central configuration for the deepfake detection system."""
    
    # Gemini API Configuration
    GEMINI_API_KEY = "AIzaSyCSz7HeKZI3YVKqortMu_MRGdXbPY06nS4"
    GEMINI_MODEL = "gemini-2.5-flash"  # Free tier model with good performance
    GEMINI_FRAMES = 7  # Send more frames for better temporal analysis
    USE_GEMINI = False  # Set to False to always use local models
    
    # Local Model configurations (fallback)
    AUDIO_MODEL = "MelodyMachine/Deepfake-audio-detection-V2"
    VIDEO_MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    
    # Video processing parameters
    NUM_FRAMES_TO_EXTRACT = 10  # Extract more frames for better accuracy
    MIN_FRAMES_REQUIRED = 3     # Minimum frames needed for valid analysis
    
    # Fusion strategy selection
    # Options: 'security_first', 'weighted', 'max', 'adaptive', 'rl_adaptive', 'advanced_rl'
    FUSION_MODE = 'rl_adaptive'  # Use simple RL with feedback learning
    
    # Fusion weights (for weighted mode & RL initial weights)
    VIDEO_WEIGHT = 0.9  # Start with 90% video weight (stronger!)
    AUDIO_WEIGHT = 0.1  # Start with 10% audio weight
    
    # RL Configuration (simple gradient-based RL - no external libraries)
    RL_LEARNING_RATE = 0.05  # How fast to adapt weights (5% per feedback)
    RL_LOAD_PREVIOUS = True  # Load previously learned weights on startup
    
    # Advanced RL Configuration (neural network-based - not used currently)
    ADVANCED_RL_ALGORITHM = 'PPO'  # 'PPO' or 'DQN'
    ADVANCED_RL_LEARNING_RATE = 0.0003  # Neural network learning rate
    ADVANCED_RL_TRAIN_STEPS = 1000  # Training steps per feedback batch
    
    # Security settings
    MISMATCH_THRESHOLD = 0.3  # Threshold for detecting modality disagreement
    
    # Thresholds
    FAKE_THRESHOLD = 0.5      # Above this is considered fake
    HIGH_CONFIDENCE_THRESHOLD = 0.75  # High confidence threshold
    
    # Temporary file settings
    TEMP_AUDIO_PATH = "temp_audio.wav"
    
    # Logging
    VERBOSE = True
    
    @classmethod
    def validate(cls):
        """Validate configuration consistency."""
        assert abs(cls.VIDEO_WEIGHT + cls.AUDIO_WEIGHT - 1.0) < 0.001, \
            "Weights must sum to 1.0"
        assert 0 < cls.FAKE_THRESHOLD < 1, "Threshold must be between 0 and 1"
        assert cls.FUSION_MODE in ['security_first', 'weighted', 'max', 'adaptive', 'rl_adaptive', 'advanced_rl'], \
            "Invalid fusion mode"
