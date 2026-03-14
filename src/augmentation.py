"""
Data augmentation for EEG and ECG 2D representations.

Applied ONLY to training data to improve generalization.
Augmentations operate on the already-transformed 2D arrays:
  - EEG: (N, 128, 9, 9)  — spatial grid
  - ECG: (N, 64, 64, 6)  — GAF+RP+MTF images
"""

import numpy as np

from src import config


def time_shift(eeg_batch, max_shift=None):
    """
    Randomly shift EEG temporal data along the time axis.

    Circular shift preserves segment length: samples shifted out at one
    end wrap around to the other.

    Args:
        eeg_batch: (N, 128, 9, 9) array.
        max_shift: Maximum shift in time steps (default: config.AUG_TIME_SHIFT_MAX).
    Returns:
        Augmented (N, 128, 9, 9) array.
    """
    if max_shift is None:
        max_shift = config.AUG_TIME_SHIFT_MAX
    if max_shift == 0:
        return eeg_batch
    result = np.empty_like(eeg_batch)
    for i in range(len(eeg_batch)):
        shift = np.random.randint(-max_shift, max_shift + 1)
        result[i] = np.roll(eeg_batch[i], shift, axis=0)
    return result


def gaussian_noise(batch, std=None):
    """
    Add Gaussian noise to a batch of 2D representations.

    Works for both EEG (N, 128, 9, 9) and ECG (N, 64, 64, 6).

    Args:
        batch: Input array.
        std: Standard deviation of noise (default: config.AUG_NOISE_STD).
    Returns:
        Augmented array, same shape.
    """
    if std is None:
        std = config.AUG_NOISE_STD
    if std == 0:
        return batch
    noise = np.random.normal(0, std, size=batch.shape).astype(batch.dtype)
    return batch + noise


def channel_dropout(eeg_batch, drop_prob=None):
    """
    Randomly zero out entire spatial grid positions (EEG channels).

    Simulates electrode failure / missing channels. Each grid position
    is independently dropped with probability `drop_prob` per sample.

    Args:
        eeg_batch: (N, 128, 9, 9) array.
        drop_prob: Per-position drop probability (default: config.AUG_CHANNEL_DROP_PROB).
    Returns:
        Augmented (N, 128, 9, 9) array.
    """
    if drop_prob is None:
        drop_prob = config.AUG_CHANNEL_DROP_PROB
    if drop_prob == 0:
        return eeg_batch

    result = eeg_batch.copy()
    grid_positions = list(config.EEG_GRID_MAP.values())

    for i in range(len(result)):
        for row, col in grid_positions:
            if np.random.random() < drop_prob:
                result[i, :, row, col] = 0.0
    return result


def amplitude_scale(batch, scale_range=None):
    """
    Randomly scale amplitude of each sample independently.

    Applies a per-sample scalar multiplier drawn from a uniform
    distribution within the given range.

    Args:
        batch: Input array (first dim = samples).
        scale_range: (low, high) tuple (default: config.AUG_AMPLITUDE_RANGE).
    Returns:
        Augmented array, same shape.
    """
    if scale_range is None:
        scale_range = config.AUG_AMPLITUDE_RANGE
    N = batch.shape[0]
    # Create a scalar per sample with shape (N, 1, 1, ...) to broadcast
    extra_dims = len(batch.shape) - 1
    shape = (N,) + (1,) * extra_dims
    scales = np.random.uniform(scale_range[0], scale_range[1], size=shape).astype(batch.dtype)
    return batch * scales


def augment_training_data(eeg_2d, ecg_2d, labels_oh,
                          augment_ratio=None, seed=None):
    """
    Generate augmented copies and append them to the original training data.

    Creates `augment_ratio` × N new augmented samples (applied to a
    random subset of originals), then concatenates with the originals.

    Args:
        eeg_2d: (N, 128, 9, 9) training EEG.
        ecg_2d: (N, 64, 64, 6) training ECG.
        labels_oh: (N, num_classes) one-hot training labels.
        augment_ratio: Fraction of data to augment (default: config.AUG_RATIO).
        seed: Random seed for reproducibility.
    Returns:
        Augmented (eeg_2d, ecg_2d, labels_oh) with extra samples appended.
    """
    if augment_ratio is None:
        augment_ratio = config.AUG_RATIO
    if augment_ratio <= 0:
        return eeg_2d, ecg_2d, labels_oh

    if seed is not None:
        np.random.seed(seed)

    N = eeg_2d.shape[0]
    n_aug = int(N * augment_ratio)
    if n_aug == 0:
        return eeg_2d, ecg_2d, labels_oh

    # Randomly sample indices to augment
    idx = np.random.choice(N, size=n_aug, replace=True)
    eeg_aug = eeg_2d[idx].copy()
    ecg_aug = ecg_2d[idx].copy()
    labels_aug = labels_oh[idx].copy()

    # Apply augmentations to the copies
    eeg_aug = time_shift(eeg_aug)
    eeg_aug = channel_dropout(eeg_aug)
    eeg_aug = gaussian_noise(eeg_aug)
    eeg_aug = amplitude_scale(eeg_aug)

    ecg_aug = gaussian_noise(ecg_aug)
    ecg_aug = amplitude_scale(ecg_aug)

    # Concatenate originals + augmented
    eeg_out = np.concatenate([eeg_2d, eeg_aug], axis=0)
    ecg_out = np.concatenate([ecg_2d, ecg_aug], axis=0)
    labels_out = np.concatenate([labels_oh, labels_aug], axis=0)

    # Shuffle the combined data
    perm = np.random.permutation(len(eeg_out))
    return eeg_out[perm], ecg_out[perm], labels_out[perm]
