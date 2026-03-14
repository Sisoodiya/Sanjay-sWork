"""
Phase 1: 1D → 2D signal transformations.
  - ECG: Gramian Angular Field (GAF), Recurrence Plot (RP), Markov Transition Field (MTF)
  - EEG: 14-channel → 9×9 spatial grid mapping

Uses CuPy for GPU acceleration when available, falls back to NumPy.
"""

import numpy as np

from src import config
from src.utils import get_xp, to_numpy, HAS_CUPY


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_minmax(x, xp):
    """Normalize signal to [-1, 1]."""
    x_min, x_max = x.min(), x.max()
    if float(x_max - x_min) < 1e-8:
        return xp.zeros_like(x)
    return 2 * (x - x_min) / (x_max - x_min) - 1


def _resize_signal(signal, target_len, xp):
    """Resize a 1D signal to target_len via linear interpolation."""
    original_len = len(signal)
    if original_len == target_len:
        return signal
    x_original = xp.linspace(0, 1, original_len)
    x_target = xp.linspace(0, 1, target_len)
    return xp.interp(x_target, x_original, signal)


def _batch_resize(signals, image_size, xp):
    """Resize (N, src_len) -> (N, image_size) with a single vectorized op (no Python loop)."""
    N, src_len = signals.shape
    if src_len == image_size:
        return signals
    # Map each target index to a float position in the source
    idx_float = xp.linspace(0, src_len - 1, image_size)
    idx_lo = xp.clip(xp.floor(idx_float).astype(xp.int32), 0, src_len - 2)
    idx_hi = idx_lo + 1
    frac = (idx_float - idx_lo)[None, :]  # (1, image_size) for broadcasting
    return signals[:, idx_lo] + frac * (signals[:, idx_hi] - signals[:, idx_lo])


# ══════════════════════════════════════════════════════════════════════════════
# ECG 2D Transforms (single-segment)
# ══════════════════════════════════════════════════════════════════════════════

def gramian_angular_field(signal_1d, image_size=None, xp=None):
    """
    Compute Gramian Angular Summation Field (GASF).

    Args:
        signal_1d: 1D array of length N.
        image_size: Output image size (default: config.ECG_IMAGE_SIZE).
        xp: Array backend (cupy or numpy).
    Returns:
        (image_size, image_size) array.
    """
    if xp is None:
        xp = get_xp()
    if image_size is None:
        image_size = config.ECG_IMAGE_SIZE
    x = _resize_signal(signal_1d, image_size, xp)
    x = _normalize_minmax(x, xp)
    x = xp.clip(x, -1, 1)
    phi = xp.arccos(x)
    gasf = xp.cos(xp.add.outer(phi, phi))
    return gasf


def recurrence_plot(signal_1d, image_size=None, threshold=None, xp=None):
    """
    Compute a Recurrence Plot from a 1D signal.

    Args:
        signal_1d: 1D array of length N.
        image_size: Output image size (default: config.ECG_IMAGE_SIZE).
        threshold: Distance threshold. If None, uses 10% of max distance.
        xp: Array backend (cupy or numpy).
    Returns:
        (image_size, image_size) array with values in [0, 1].
    """
    if xp is None:
        xp = get_xp()
    if image_size is None:
        image_size = config.ECG_IMAGE_SIZE
    x = _resize_signal(signal_1d, image_size, xp)
    dist_matrix = xp.abs(x[:, None] - x[None, :])
    max_dist = dist_matrix.max()
    if float(max_dist) > 1e-8:
        dist_matrix = dist_matrix / max_dist
    if threshold is None:
        threshold = 0.1
    rp = (dist_matrix < threshold).astype(xp.float32)
    return rp


def markov_transition_field(signal_1d, image_size=None, n_bins=8, xp=None):
    """
    Compute a Markov Transition Field (MTF) from a 1D signal.

    Uses fully vectorized indexing instead of nested loops.

    Args:
        signal_1d: 1D array of length N.
        image_size: Output image size (default: config.ECG_IMAGE_SIZE).
        n_bins: Number of quantile bins.
        xp: Array backend (cupy or numpy).
    Returns:
        (image_size, image_size) array.
    """
    if xp is None:
        xp = get_xp()
    if image_size is None:
        image_size = config.ECG_IMAGE_SIZE
    x = _resize_signal(signal_1d, image_size, xp)

    x_min, x_max = float(x.min()), float(x.max())
    if x_max - x_min < 1e-8:
        return xp.zeros((image_size, image_size), dtype=xp.float32)
    bins = xp.linspace(x_min, x_max, n_bins + 1)
    binned = xp.clip(xp.digitize(x, bins) - 1, 0, n_bins - 1)

    transition_matrix = xp.zeros((n_bins, n_bins), dtype=xp.float32)
    from_bins = binned[:-1]
    to_bins = binned[1:]
    if HAS_CUPY:
        xp.add.at(transition_matrix, (from_bins, to_bins), 1)
    else:
        np.add.at(transition_matrix, (from_bins.astype(int), to_bins.astype(int)), 1)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums = xp.where(row_sums == 0, 1, row_sums)
    transition_matrix = transition_matrix / row_sums

    mtf = transition_matrix[binned[:, None], binned[None, :]]
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
        (image_size, image_size, 6) numpy array.
    """
    if image_size is None:
        image_size = config.ECG_IMAGE_SIZE
    xp = get_xp()
    seg = xp.asarray(ecg_segment)
    channels = []
    for ch in range(seg.shape[1]):
        sig = seg[:, ch]
        channels.append(gramian_angular_field(sig, image_size, xp))
        channels.append(recurrence_plot(sig, image_size, xp=xp))
        channels.append(markov_transition_field(sig, image_size, xp=xp))
    result = xp.stack(channels, axis=-1)
    return to_numpy(result)


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
# Batch Transforms (GPU-accelerated)
# ══════════════════════════════════════════════════════════════════════════════

def _batch_gaf(signals, image_size, xp):
    """Batched GAF: (N, signal_len) -> (N, image_size, image_size)."""
    resized = _batch_resize(signals, image_size, xp)
    mins = resized.min(axis=1, keepdims=True)
    maxs = resized.max(axis=1, keepdims=True)
    rng = xp.where(maxs - mins < 1e-8, 1.0, maxs - mins)
    normed = xp.clip(2 * (resized - mins) / rng - 1, -1, 1)
    phi = xp.arccos(normed)  # (N, image_size)
    return xp.cos(phi[:, :, None] + phi[:, None, :])


def _batch_rp(signals, image_size, threshold, xp):
    """Batched RP: (N, signal_len) -> (N, image_size, image_size)."""
    resized = _batch_resize(signals, image_size, xp)
    N = resized.shape[0]
    dist = xp.abs(resized[:, :, None] - resized[:, None, :])  # (N, S, S)
    max_dist = xp.where(
        dist.reshape(N, -1).max(axis=1)[:, None, None] < 1e-8,
        1.0,
        dist.reshape(N, -1).max(axis=1)[:, None, None],
    )
    return (dist / max_dist < threshold).astype(xp.float32)


def _batch_mtf(signals, image_size, n_bins, xp):
    """Batched MTF: (N, signal_len) -> (N, image_size, image_size).

    Fully vectorized — no Python loop over samples.
    """
    resized = _batch_resize(signals, image_size, xp)
    N, S = resized.shape

    # Per-sample min/max: (N, 1)
    x_mins = resized.min(axis=1, keepdims=True)
    x_maxs = resized.max(axis=1, keepdims=True)
    rng = x_maxs - x_mins

    # Mask for constant signals (range ~ 0)
    valid = (rng.squeeze(1) > 1e-8)  # (N,)

    # Normalize to [0, 1] then quantize to bin indices
    safe_rng = xp.where(rng < 1e-8, 1.0, rng)
    normed = (resized - x_mins) / safe_rng  # (N, S), values in [0, 1]
    # Map to bin indices [0, n_bins-1]
    binned = xp.clip((normed * n_bins).astype(xp.int32), 0, n_bins - 1)  # (N, S)

    # Build transition matrices vectorized using one-hot scatter
    from_bins = binned[:, :-1]  # (N, S-1)
    to_bins = binned[:, 1:]     # (N, S-1)

    # Flatten to 1D index: sample * n_bins * n_bins + from_bin * n_bins + to_bin
    sample_idx = xp.arange(N)[:, None] * (n_bins * n_bins)  # (N, 1)
    flat_idx = sample_idx + from_bins * n_bins + to_bins      # (N, S-1)
    flat_idx = flat_idx.reshape(-1)

    tm_flat = xp.zeros(N * n_bins * n_bins, dtype=xp.float32)
    if HAS_CUPY:
        xp.add.at(tm_flat, flat_idx, 1)
    else:
        np.add.at(tm_flat, flat_idx.astype(int), 1)
    tm = tm_flat.reshape(N, n_bins, n_bins)  # (N, n_bins, n_bins)

    # Row-normalize transition matrices
    row_sums = tm.sum(axis=2, keepdims=True)
    row_sums = xp.where(row_sums == 0, 1.0, row_sums)
    tm = tm / row_sums  # (N, n_bins, n_bins)

    # Build MTF images via advanced indexing: MTF[i,j] = tm[binned[i], binned[j]]
    row_idx = binned[:, :, None].repeat(S, axis=2)   # (N, S, S)
    col_idx = binned[:, None, :].repeat(S, axis=1)   # (N, S, S)
    batch_idx = xp.arange(N)[:, None, None]           # (N, 1, 1)
    results = tm[batch_idx, row_idx, col_idx]         # (N, S, S)

    # Zero out results for constant-signal samples
    results = results * valid[:, None, None].astype(xp.float32)

    return results


def transform_ecg_batch(ecg_segments, image_size=None):
    """
    Apply 2D transforms to a batch of ECG segments using GPU if available.

    Processes in chunks of config.TRANSFORM_BATCH_SIZE to cap peak RAM
    usage (important on Colab with ~12GB system RAM).

    Args:
        ecg_segments: (N, 256, 2) array.
        image_size: Output spatial resolution.
    Returns:
        (N, image_size, image_size, 6) numpy array.
    """
    if image_size is None:
        image_size = config.ECG_IMAGE_SIZE
    N = ecg_segments.shape[0]
    chunk_size = config.TRANSFORM_BATCH_SIZE

    # Pre-allocate output array
    result = np.empty((N, image_size, image_size, 6), dtype=np.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        xp = get_xp()
        segs = xp.asarray(ecg_segments[start:end])

        all_channels = []
        for ch in range(segs.shape[2]):
            ch_data = segs[:, :, ch]  # (chunk, 256)
            all_channels.append(_batch_gaf(ch_data, image_size, xp))
            all_channels.append(_batch_rp(ch_data, image_size, 0.1, xp))
            all_channels.append(_batch_mtf(ch_data, image_size, 8, xp))

        chunk_result = to_numpy(xp.stack(all_channels, axis=-1)).astype(np.float32)
        result[start:end] = chunk_result
        del segs, all_channels, chunk_result  # Free chunk memory

    return result


def transform_eeg_batch(eeg_segments, grid_size=None):
    """
    Apply 2D grid transform to a batch of EEG segments.

    Args:
        eeg_segments: (N, 128, 14) array.
        grid_size: Grid dimension.
    Returns:
        (N, 128, grid_size, grid_size) numpy array.
    """
    if grid_size is None:
        grid_size = config.EEG_GRID_SIZE
    segs = np.asarray(eeg_segments)
    N, T, _ = segs.shape
    grid = np.zeros((N, T, grid_size, grid_size), dtype=np.float32)
    for ch_idx, (row, col) in config.EEG_GRID_MAP.items():
        grid[:, :, row, col] = segs[:, :, ch_idx]
    return grid


if __name__ == "__main__":
    print(f"CuPy available: {HAS_CUPY}")

    print("Testing ECG 2D transforms...")
    ecg_seg = np.random.randn(256, 2)
    ecg_2d = ecg_to_2d(ecg_seg)
    print(f"  ECG input: {ecg_seg.shape} -> ECG 2D output: {ecg_2d.shape}")
    assert ecg_2d.shape == (64, 64, 6), f"Expected (64,64,6), got {ecg_2d.shape}"

    print("Testing EEG 2D grid transform...")
    eeg_seg = np.random.randn(128, 14)
    eeg_2d = eeg_to_2d_grid(eeg_seg)
    print(f"  EEG input: {eeg_seg.shape} -> EEG 2D output: {eeg_2d.shape}")
    assert eeg_2d.shape == (128, 9, 9), f"Expected (128,9,9), got {eeg_2d.shape}"

    print("Testing batch transforms...")
    ecg_batch = np.random.randn(5, 256, 2)
    ecg_batch_2d = transform_ecg_batch(ecg_batch)
    print(f"  ECG batch: {ecg_batch.shape} -> {ecg_batch_2d.shape}")

    eeg_batch = np.random.randn(5, 128, 14)
    eeg_batch_2d = transform_eeg_batch(eeg_batch)
    print(f"  EEG batch: {eeg_batch.shape} -> {eeg_batch_2d.shape}")

    print("All transform tests passed!")
