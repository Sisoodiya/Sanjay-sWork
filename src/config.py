"""
Configuration for Emotion Recognition with Enhanced Att-1DCNN-GRU + TACO Cross-Attention.
Dataset: DREAMER (23 subjects, 18 trials, 14-ch EEG @ 128Hz, 2-ch ECG @ 256Hz)
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "DREAMER.mat")
SAVE_DIR = os.path.join(BASE_DIR, "results")

# ── Dataset ───────────────────────────────────────────────────────────────────
NUM_SUBJECTS = 23
NUM_TRIALS = 18
EEG_CHANNELS = 14
ECG_CHANNELS = 2
EEG_SR = 128       # EEG sampling rate (Hz)
ECG_SR = 256       # ECG sampling rate (Hz)

EEG_ELECTRODE_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

# ── Signal Extraction ────────────────────────────────────────────────────────
LAST_SECONDS = 60          # Extract last N seconds of each trial
SEGMENT_LENGTH_SEC = 1     # 1-second segments
EEG_SEGMENT_SAMPLES = EEG_SR * SEGMENT_LENGTH_SEC   # 128
ECG_SEGMENT_SAMPLES = ECG_SR * SEGMENT_LENGTH_SEC   # 256

# ── Label Mapping ─────────────────────────────────────────────────────────────
NUM_CLASSES = 3
LABEL_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
# Class 0 (Low):    scores 1, 2
# Class 1 (Medium): score  3
# Class 2 (High):   scores 4, 5

TARGETS = ["valence", "arousal", "dominance"]

# ── EEG 2D Grid Mapping ──────────────────────────────────────────────────────
# Maps 14 Emotiv EPOC channels to positions on a 9×9 grid
# reflecting the physical topology of the scalp
EEG_GRID_SIZE = 9
EEG_GRID_MAP = {
    # channel_index: (row, col) in 9×9 grid
    0:  (0, 3),   # AF3
    13: (0, 5),   # AF4
    1:  (1, 1),   # F7
    2:  (1, 3),   # F3
    11: (1, 5),   # F4
    12: (1, 7),   # F8
    3:  (2, 2),   # FC5
    10: (2, 6),   # FC6
    4:  (3, 1),   # T7
    9:  (3, 7),   # T8
    5:  (5, 1),   # P7
    8:  (5, 7),   # P8
    6:  (6, 3),   # O1
    7:  (6, 5),   # O2
}

# ── ECG 2D Image ──────────────────────────────────────────────────────────────
ECG_IMAGE_SIZE = 64    # Output resolution for GAF/RP/MTF images
ECG_IMAGE_CHANNELS = 6 # 2 ECG channels × 3 transforms (GAF, RP, MTF)

# ── Transformer Config ────────────────────────────────────────────────────────
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_TRANSFORMER_LAYERS = 2
DROPOUT_RATE = 0.3

# ── ECG Patch Embedding ──────────────────────────────────────────────────────
ECG_PATCH_SIZE = 8  # 64/8 = 8×8 = 64 patches

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
RANDOM_SEED = 42
