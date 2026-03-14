"""
Shared utility functions for the emotion recognition pipeline.
"""

import os
import random
import numpy as np


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
