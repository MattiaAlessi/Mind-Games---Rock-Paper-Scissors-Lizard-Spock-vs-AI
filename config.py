"""
Configuration module for Mind Games - Rock Paper Scissors Lizard Spock
Centralizes all constants and settings for easy maintenance
"""

import os
from enum import Enum

# ============================================================================
# PATHS & FILES
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
AUDIO_DIR = os.path.join(PROJECT_ROOT, 'audio')

# Model files
GESTURE_MODEL_PATH = os.path.join(MODELS_DIR, 'gesture_classifier.pkl')
GESTURE_SCALER_PATH = os.path.join(MODELS_DIR, 'gesture_scaler.pkl')
LSTM_PREDICTOR_PATH = os.path.join(MODELS_DIR, 'sequence_predictor.h5')

# Audio files
SOUND_COUNTDOWN = os.path.join(AUDIO_DIR, 'countdown.wav')
SOUND_WIN = os.path.join(AUDIO_DIR, 'win.wav')
SOUND_LOSE = os.path.join(AUDIO_DIR, 'lose.wav')
SOUND_DRAW = os.path.join(AUDIO_DIR, 'draw.wav')

# Create directories if missing
for d in [DATA_DIR, MODELS_DIR, AUDIO_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# GESTURES
# ============================================================================
class Gesture(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    LIZARD = 3
    SPOCK = 4

GESTURE_NAMES = {
    Gesture.ROCK: 'Sasso',
    Gesture.PAPER: 'Carta',
    Gesture.SCISSORS: 'Forbice',
    Gesture.LIZARD: 'Lizard',
    Gesture.SPOCK: 'Spock'
}

GESTURE_EMOJIS = {
    Gesture.ROCK: '✊',
    Gesture.PAPER: '✋',
    Gesture.SCISSORS: '✌️',
    Gesture.LIZARD: '🦎',
    Gesture.SPOCK: '🖖'
}

# ============================================================================
# GAME LOGIC (RPSLS RULES)
# ============================================================================
WINS = {
    Gesture.ROCK: [Gesture.SCISSORS, Gesture.LIZARD],
    Gesture.PAPER: [Gesture.ROCK, Gesture.SPOCK],
    Gesture.SCISSORS: [Gesture.PAPER, Gesture.LIZARD],
    Gesture.LIZARD: [Gesture.PAPER, Gesture.SPOCK],
    Gesture.SPOCK: [Gesture.SCISSORS, Gesture.ROCK]
}

# ============================================================================
# MEDIAPIPE & HAND TRACKING
# ============================================================================
MP_HANDS_MAX_HANDS = 1
MP_HANDS_MIN_CONFIDENCE = 0.7
MP_HANDS_TRACKING_CONFIDENCE = 0.5

# Hand landmarks (21 points)
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 3  # x, y, z
NUM_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 63

# Hand connections for skeleton drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17)                                   # Palm connection
]

# Smoothing parameters
SMOOTHING_WINDOW = 5  # deque size for temporal smoothing

# ============================================================================
# LIGHTGBM CLASSIFIER
# ============================================================================
LGBM_PARAMS = {
    'num_leaves': 31,
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'random_state': 42,
    'verbose': -1,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 0.1   # L2 regularization
}

LGBM_CV_FOLDS = 5
LGBM_MIN_SAMPLES_PER_CLASS = 50

# ============================================================================
# LSTM SEQUENCE PREDICTOR
# ============================================================================
LSTM_SEQUENCE_LENGTH = 10  # Context window
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 50
LSTM_VALIDATION_SPLIT = 0.2

LSTM_LAYERS = {
    'lstm_units': 64,
    'dropout_rate': 0.3,
    'dense_units': 32
}

# ============================================================================
# WEBCAM & VIDEO PROCESSING
# ============================================================================
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
WEBCAM_FPS = 30
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame for performance

# ============================================================================
# PYGAME INTERFACE
# ============================================================================
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCREEN_FPS = 60
BACKGROUND_COLOR = (10, 10, 20)  # Dark blue-black

# Layout divisions
WEBCAM_AREA_WIDTH = int(SCREEN_WIDTH * 0.6)
STATS_AREA_WIDTH = SCREEN_WIDTH - WEBCAM_AREA_WIDTH

# Colors (BGR for OpenCV, RGB for Pygame)
COLOR_PRIMARY = (0, 200, 255)    # Cyan
COLOR_SUCCESS = (0, 255, 100)    # Green
COLOR_DANGER = (0, 50, 255)      # Red
COLOR_WARNING = (0, 165, 255)    # Orange
COLOR_TEXT = (255, 255, 255)     # White
COLOR_TEXT_DIM = (150, 150, 150) # Gray

# Fonts
FONT_LARGE = 48
FONT_MEDIUM = 32
FONT_SMALL = 24

# ============================================================================
# GAME PARAMETERS
# ============================================================================
COUNTDOWN_DURATION = 3  # seconds
MOVE_DISPLAY_DURATION = 2  # seconds to show result
SEQUENCE_HISTORY_LENGTH = 10
MIN_CONFIDENCE_TO_CLASSIFY = 0.6

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
AUGMENTATION_NOISE_STD = 0.02
AUGMENTATION_ROTATION_RANGE = 10  # degrees
AUGMENTATION_SCALE_RANGE = 0.1    # ±10%

# ============================================================================
# LOGGING & DEBUG
# ============================================================================
DEBUG_MODE = False
LOG_PREDICTIONS = False
SHOW_CONFIDENCE_SCORES = True
SHOW_SKELETON = True
