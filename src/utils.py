"""
Shared utility functions for the emotion recognition pipeline.
"""

import os
import random
import numpy as np

# ── CuPy / NumPy backend ────────────────────────────────────────────────────
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def get_xp():
    """Return cupy if available (and GPU present), else numpy."""
    if HAS_CUPY and cp.cuda.runtime.getDeviceCount() > 0:
        return cp
    return np


def to_numpy(x):
    """Convert cupy array to numpy if needed."""
    if HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def set_seed(seed=42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_class_weights(labels):
    """
    Compute class weights to handle imbalance.
    Args:
        labels: 1D array of integer class labels.
    Returns:
        Dictionary mapping class index → weight.
    """
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(counts)
    weights = {}
    for cls, count in counts.items():
        weights[cls] = total / (n_classes * count)
    return weights


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
