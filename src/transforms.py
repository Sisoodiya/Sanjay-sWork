"""
Phase 1: 1D → 2D signal transformations.
  - ECG: Gramian Angular Field (GAF), Recurrence Plot (RP), Markov Transition Field (MTF)
  - EEG: 14-channel → 9×9 spatial grid mapping
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from src import config


# ══════════════════════════════════════════════════════════════════════════════
# ECG 2D Transforms
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_minmax(x):
    """Normalize signal to [-1, 1]."""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return 2 * (x - x_min) / (x_max - x_min) - 1


def _resize_signal(signal, target_len):
    """Downsample/upsample signal to target_len using linear interpolation."""
    original_len = len(signal)
    if original_len == target_len:
        return signal
    x_original = np.linspace(0, 1, original_len)
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_original, signal)


def gramian_angular_field(signal_1d, image_size=None):
    """
    Compute Gramian Angular Summation Field (GASF).

    The signal is normalized to [-1, 1], converted to angular representation
    via arccos, then the outer sum of angles forms the 2D image.

    Args:
        signal_1d: 1D array of length N.
        image_size: Output image size (default: config.ECG_IMAGE_SIZE).
    Returns:
        (image_size, image_size) array.
    """
    if image_size is None:
        image_size = config.ECG_IMAGE_SIZE
    # Resize to target image size
    x = _resize_signal(signal_1d, image_size)
    # Normalize to [-1, 1]
    x = _normalize_minmax(x)
    # Clamp for numerical safety
    x = np.clip(x, -1, 1)
    # Angular representation
    phi = np.arccos(x)
    # GASF: cos(phi_i + phi_j)
    gasf = np.cos(np.add.outer(phi, phi))
    return gasf


def recurrence_plot(signal_1d, image_size=None, threshold=None):
    """
    Compute a Recurrence Plot from a 1D signal.

    Uses time-delay embedding (tau=1, dim=1 for simplicity) and a distance
    threshold to create a binary recurrence matrix.

    Args:
        signal_1d: 1D array of length N.
        image_size: Output image size (default: config.ECG_IMAGE_SIZE).
        threshold: Distance threshold. If None, uses 10% of max distance.
    Returns:
        (image_size, image_size) array with values in [0, 1].
    """
    if image_size is None:
        image_size = config.ECG_IMAGE_SIZE
    x = _resize_signal(signal_1d, image_size)
    # Compute pairwise distances
    x_col = x.reshape(-1, 1)
    dist_matrix = squareform(pdist(x_col, metric="euclidean"))
    # Normalize distances
    max_dist = dist_matrix.max()
    if max_dist > 1e-8:
        dist_matrix = dist_matrix / max_dist
    if threshold is None:
        threshold = 0.1
    # Binary recurrence: 1 if distance < threshold, else 0
    rp = (dist_matrix < threshold).astype(np.float64)
    return rp


def markov_transition_field(signal_1d, image_size=None, n_bins=8):
    """
    Compute a Markov Transition Field (MTF) from a 1D signal.

    Quantizes the signal into bins, computes the transition probability matrix,
    and maps time-indexed transitions back to a 2D image.

    Args:
        signal_1d: 1D array of length N.
        image_size: Output image size (default: config.ECG_IMAGE_SIZE).
        n_bins: Number of quantile bins.
    Returns:
        (image_size, image_size) array.
    """
    if image_size is None:
        image_size = config.ECG_IMAGE_SIZE
    x = _resize_signal(signal_1d, image_size)

    # Quantize into bins using quantile boundaries
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return np.zeros((image_size, image_size))
    bins = np.linspace(x_min, x_max, n_bins + 1)
    # Digitize: assign each sample to a bin (0 to n_bins-1)
    binned = np.clip(np.digitize(x, bins) - 1, 0, n_bins - 1)

    # Compute transition matrix
    transition_matrix = np.zeros((n_bins, n_bins))
    for i in range(len(binned) - 1):
        transition_matrix[binned[i], binned[i + 1]] += 1
    # Normalize rows to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums

    # Build MTF: M[i,j] = transition probability from bin(x[i]) to bin(x[j])
    mtf = np.zeros((image_size, image_size))
    for i in range(image_size):
        for j in range(image_size):
            mtf[i, j] = transition_matrix[binned[i], binned[j]]

    return mtf


def ecg_to_2d(ecg_segment, image_size=None):
    """
    Convert a 1-second ECG segment to a multi-channel 2D image.

    Applies GAF, RP, and MTF to each of the 2 ECG channels,
    producing 6 image channels total.

    Args:
        ecg_segment: (256, 2) array — one 1s ECG segment.
        image_size: Output spatial resolution.
    Returns:
        (image_size, image_size, 6) array.
    """
    if image_size is None:
        image_size = config.ECG_IMAGE_SIZE
    channels = []
    for ch in range(ecg_segment.shape[1]):
        sig = ecg_segment[:, ch]
        channels.append(gramian_angular_field(sig, image_size))
        channels.append(recurrence_plot(sig, image_size))
        channels.append(markov_transition_field(sig, image_size))
    # Stack: (image_size, image_size, 6)
    return np.stack(channels, axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# EEG 2D Grid Transform
# ══════════════════════════════════════════════════════════════════════════════

def eeg_to_2d_grid(eeg_segment, grid_size=None):
    """
    Map a 14-channel EEG segment onto a 2D spatial grid.

    Each time step's 14 channel values are placed at their corresponding
    positions in a 9×9 grid reflecting the Emotiv EPOC electrode layout.
    Unoccupied positions are zero.

    Args:
        eeg_segment: (128, 14) array — one 1s EEG segment.
        grid_size: Grid dimension (default: config.EEG_GRID_SIZE).
    Returns:
        (128, grid_size, grid_size) array — 3D spatial-temporal tensor.
    """
    if grid_size is None:
        grid_size = config.EEG_GRID_SIZE
    n_timesteps = eeg_segment.shape[0]
    grid = np.zeros((n_timesteps, grid_size, grid_size), dtype=np.float32)

    for ch_idx, (row, col) in config.EEG_GRID_MAP.items():
        grid[:, row, col] = eeg_segment[:, ch_idx]

    return grid


# ══════════════════════════════════════════════════════════════════════════════
# Batch Transform
# ══════════════════════════════════════════════════════════════════════════════

def transform_ecg_batch(ecg_segments, image_size=None):
    """
    Apply 2D transform to a batch of ECG segments.

    Args:
        ecg_segments: (N, 256, 2) array.
        image_size: Output spatial resolution.
    Returns:
        (N, image_size, image_size, 6) array.
    """
    if image_size is None:
        image_size = config.ECG_IMAGE_SIZE
    results = []
    for i in range(len(ecg_segments)):
        results.append(ecg_to_2d(ecg_segments[i], image_size))
    return np.array(results, dtype=np.float32)


def transform_eeg_batch(eeg_segments, grid_size=None):
    """
    Apply 2D grid transform to a batch of EEG segments.

    Args:
        eeg_segments: (N, 128, 14) array.
        grid_size: Grid dimension.
    Returns:
        (N, 128, grid_size, grid_size) array.
    """
    if grid_size is None:
        grid_size = config.EEG_GRID_SIZE
    results = []
    for i in range(len(eeg_segments)):
        results.append(eeg_to_2d_grid(eeg_segments[i], grid_size))
    return np.array(results, dtype=np.float32)


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing ECG 2D transforms...")
    ecg_seg = np.random.randn(256, 2)
    ecg_2d = ecg_to_2d(ecg_seg)
    print(f"  ECG input: {ecg_seg.shape} → ECG 2D output: {ecg_2d.shape}")
    assert ecg_2d.shape == (64, 64, 6), f"Expected (64,64,6), got {ecg_2d.shape}"

    print("Testing EEG 2D grid transform...")
    eeg_seg = np.random.randn(128, 14)
    eeg_2d = eeg_to_2d_grid(eeg_seg)
    print(f"  EEG input: {eeg_seg.shape} → EEG 2D output: {eeg_2d.shape}")
    assert eeg_2d.shape == (128, 9, 9), f"Expected (128,9,9), got {eeg_2d.shape}"

    print("Testing batch transforms...")
    ecg_batch = np.random.randn(5, 256, 2)
    ecg_batch_2d = transform_ecg_batch(ecg_batch)
    print(f"  ECG batch: {ecg_batch.shape} → {ecg_batch_2d.shape}")

    eeg_batch = np.random.randn(5, 128, 14)
    eeg_batch_2d = transform_eeg_batch(eeg_batch)
    print(f"  EEG batch: {eeg_batch.shape} → {eeg_batch_2d.shape}")

    print("All transform tests passed!")
