"""
Configuration module for deepfake detection system.
"""

class Config:
    """Central configuration for the deepfake detection system."""
    
    # ─── Video Model (GenConViT) ─────────────────────────────────
    
    # GenConViT (CVIT) Configuration
    CVIT_NET = 'genconvit'                         # 'genconvit' (both ED+VAE), 'ed', or 'vae'
    CVIT_ED_WEIGHT = 'genconvit_ed_inference'      # ED weight filename (without .pth)
    CVIT_VAE_WEIGHT = 'genconvit_vae_inference'    # VAE weight filename (without .pth)
    CVIT_FP16 = False                              # Use half precision (faster on GPU)
    
    # Gemini API Configuration (optional, disabled by default)
    GEMINI_API_KEY = ""
    GEMINI_MODEL = "gemini-2.5-flash"
    GEMINI_FRAMES = 7
    USE_GEMINI = False
    
    # Audio Model — Wav2Vec2 deepfake voice detector
    AUDIO_MODEL = "garystafford/wav2vec2-deepfake-voice-detector"
    
    # Video processing parameters
    NUM_FRAMES_TO_EXTRACT = 10  # Extract more frames for better accuracy
    MIN_FRAMES_REQUIRED = 3     # Minimum frames needed for valid analysis
    
    # Fusion strategy selection
    # Options: 'security_first', 'weighted', 'max', 'adaptive', 'rl_adaptive', 'advanced_rl'
    FUSION_MODE = 'rl_adaptive'  # Use simple RL with feedback learning
    
    # Fusion weights — GenConViT is primary, audio is supplementary
    VIDEO_WEIGHT = 0.95  # GenConViT video model (primary)
    AUDIO_WEIGHT = 0.05  # HuggingFace audio model (secondary)
    
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
