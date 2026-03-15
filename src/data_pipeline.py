"""
Memory-efficient data pipeline for LOSOCV training.

Replaces monolithic numpy arrays with per-subject disk caching and
tf.data streaming. Peak RAM drops from >4 GB to <1 GB, enabling
training on Google Colab free tier (~12.7 GB system RAM).

Data flow:
    1. build_dataset() produces preprocessed segments (once)
    2. save_subject_files() pre-computes 2D transforms and saves
       per-subject .npy files to data/subjects/
    3. make_training_dataset() creates a tf.data.Dataset generator
       that loads one subject at a time (~76 MB) with inline augmentation
    4. load_eval_data() loads val/test subjects as small numpy arrays
"""

import os
import gc

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from src import config
from src.transforms import transform_eeg_batch, transform_ecg_batch

SUBJECTS_DIR = config.SUBJECTS_CACHE_DIR


# ── Per-Subject Disk Cache ──────────────────────────────────────────────────

def save_subject_files(dataset, subjects_dir=SUBJECTS_DIR):
    """
    Pre-compute 2D transforms for each subject and save as .npy files.

    Layout:
        data/subjects/subject_00/eeg_2d.npy          (N, 128, 9, 9) float16
        data/subjects/subject_00/ecg_2d.npy          (N, 64, 64, 6) float16
        data/subjects/subject_00/labels_valence.npy  (N,) int
        data/subjects/subject_00/labels_arousal.npy  (N,) int
        data/subjects/subject_00/labels_dominance.npy (N,) int

    Args:
        dataset: List of 23 dicts from build_dataset(), each with
                 'eeg_segments', 'ecg_segments', 'labels_valence', etc.
        subjects_dir: Output directory.
    """
    os.makedirs(subjects_dir, exist_ok=True)
    print(f"Saving per-subject 2D cache to {subjects_dir}...")

    for i, subj in enumerate(dataset):
        subj_dir = os.path.join(subjects_dir, f"subject_{i:02d}")
        os.makedirs(subj_dir, exist_ok=True)

        eeg_2d = transform_eeg_batch(subj["eeg_segments"])
        np.save(os.path.join(subj_dir, "eeg_2d.npy"), eeg_2d)
        del eeg_2d

        ecg_2d = transform_ecg_batch(subj["ecg_segments"])
        np.save(os.path.join(subj_dir, "ecg_2d.npy"), ecg_2d)
        del ecg_2d

        for target in config.TARGETS:
            label_key = f"labels_{target}"
            np.save(os.path.join(subj_dir, f"{label_key}.npy"), subj[label_key])

        n_segs = len(subj["labels_valence"])
        print(f"  Subject {i:2d}: {n_segs} segments saved")
        gc.collect()

    print(f"Per-subject 2D cache complete ({len(dataset)} subjects).")


def subjects_cache_exists(subjects_dir=SUBJECTS_DIR):
    """Check if the per-subject 2D cache is complete."""
    if not os.path.isdir(subjects_dir):
        return False
    required = ["eeg_2d.npy", "ecg_2d.npy",
                 "labels_valence.npy", "labels_arousal.npy",
                 "labels_dominance.npy"]
    for i in range(config.NUM_SUBJECTS):
        subj_dir = os.path.join(subjects_dir, f"subject_{i:02d}")
        for f in required:
            if not os.path.isfile(os.path.join(subj_dir, f)):
                return False
    return True


# ── Per-Sample Augmentation ─────────────────────────────────────────────────

def _augment_eeg_sample(eeg, rng):
    """Augment a single EEG 2D sample (128, 9, 9) float16."""
    result = eeg.copy()
    # Time shift
    shift = rng.randint(-config.AUG_TIME_SHIFT_MAX, config.AUG_TIME_SHIFT_MAX + 1)
    if shift != 0:
        result = np.roll(result, shift, axis=0)
    # Channel dropout
    for row, col in config.EEG_GRID_MAP.values():
        if rng.random() < config.AUG_CHANNEL_DROP_PROB:
            result[:, row, col] = 0.0
    # Gaussian noise
    result = result + rng.normal(0, config.AUG_NOISE_STD,
                                  result.shape).astype(result.dtype)
    # Amplitude scale
    scale = rng.uniform(*config.AUG_AMPLITUDE_RANGE)
    result = (result * np.float16(scale))
    return result


def _augment_ecg_sample(ecg, rng):
    """Augment a single ECG 2D sample (64, 64, 6) float16."""
    result = ecg.copy()
    # Gaussian noise
    result = result + rng.normal(0, config.AUG_NOISE_STD,
                                  result.shape).astype(result.dtype)
    # Amplitude scale
    scale = rng.uniform(*config.AUG_AMPLITUDE_RANGE)
    result = (result * np.float16(scale))
    return result


# ── tf.data Generator ───────────────────────────────────────────────────────

def _training_generator(train_subject_indices, target, subjects_dir,
                         aug_ratio):
    """
    Generator yielding ((eeg, ecg), label_oh) one sample at a time.

    Loads one subject from disk at a time (~76 MB), yields all its
    samples (with inline augmentation), then frees memory.

    Subject order and within-subject sample order are shuffled each
    time the generator is called (each epoch).
    """
    rng = np.random.RandomState(config.RANDOM_SEED)
    shuffled_subjects = list(rng.permutation(train_subject_indices))

    for subj_idx in shuffled_subjects:
        subj_dir = os.path.join(subjects_dir, f"subject_{subj_idx:02d}")

        eeg_2d = np.load(os.path.join(subj_dir, "eeg_2d.npy"))
        ecg_2d = np.load(os.path.join(subj_dir, "ecg_2d.npy"))
        labels = np.load(os.path.join(subj_dir, f"labels_{target}.npy"))
        labels_oh = to_categorical(labels, config.NUM_CLASSES).astype(np.float32)

        n_subj = len(labels)
        inner_perm = rng.permutation(n_subj)

        for i in inner_perm:
            yield (eeg_2d[i], ecg_2d[i]), labels_oh[i]

            # With probability aug_ratio, yield an augmented copy
            if aug_ratio > 0 and rng.random() < aug_ratio:
                aug_eeg = _augment_eeg_sample(eeg_2d[i], rng)
                aug_ecg = _augment_ecg_sample(ecg_2d[i], rng)
                yield (aug_eeg, aug_ecg), labels_oh[i]

        del eeg_2d, ecg_2d, labels, labels_oh
        gc.collect()


def make_training_dataset(train_subject_indices, target, batch_size=None,
                           subjects_dir=SUBJECTS_DIR, aug_ratio=None):
    """
    Create a tf.data.Dataset that streams training data from disk.

    Args:
        train_subject_indices: List of subject indices for training.
        target: 'valence', 'arousal', or 'dominance'.
        batch_size: Batch size (default: config.BATCH_SIZE).
        subjects_dir: Path to per-subject cache directory.
        aug_ratio: Augmentation ratio (default: config.AUG_RATIO).
    Returns:
        tf.data.Dataset yielding ((eeg_batch, ecg_batch), labels_batch).
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if aug_ratio is None:
        aug_ratio = config.AUG_RATIO

    output_signature = (
        (tf.TensorSpec(shape=(128, 9, 9), dtype=tf.float16),
         tf.TensorSpec(shape=(64, 64, 6), dtype=tf.float16)),
        tf.TensorSpec(shape=(config.NUM_CLASSES,), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: _training_generator(
            train_subject_indices, target, subjects_dir, aug_ratio),
        output_signature=output_signature,
    )

    # Shuffle buffer mixes samples across subject boundaries.
    # 2000 samples * ~70 KB = ~140 MB — good mixing for ~1080 segs/subject.
    ds = ds.shuffle(buffer_size=config.SHUFFLE_BUFFER_SIZE,
                    reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ── Eval Data Loading ───────────────────────────────────────────────────────

def load_eval_data(subject_idx, target, subjects_dir=SUBJECTS_DIR):
    """
    Load one subject's 2D-transformed data for validation or testing.

    Returns:
        (eeg_2d, ecg_2d, labels, labels_oh)
    """
    subj_dir = os.path.join(subjects_dir, f"subject_{subject_idx:02d}")
    eeg_2d = np.load(os.path.join(subj_dir, "eeg_2d.npy"))
    ecg_2d = np.load(os.path.join(subj_dir, "ecg_2d.npy"))
    labels = np.load(os.path.join(subj_dir, f"labels_{target}.npy"))
    labels_oh = to_categorical(labels, config.NUM_CLASSES).astype(np.float32)
    return eeg_2d, ecg_2d, labels, labels_oh


# ── Utilities ───────────────────────────────────────────────────────────────

def count_training_samples(train_subject_indices, target,
                            subjects_dir=SUBJECTS_DIR, aug_ratio=None):
    """Count expected training samples (raw + augmented) for LR schedule."""
    if aug_ratio is None:
        aug_ratio = config.AUG_RATIO
    n_raw = 0
    for subj_idx in train_subject_indices:
        subj_dir = os.path.join(subjects_dir, f"subject_{subj_idx:02d}")
        labels = np.load(os.path.join(subj_dir, f"labels_{target}.npy"))
        n_raw += len(labels)
    return int(n_raw * (1 + aug_ratio))


def get_training_labels(train_subject_indices, target,
                         subjects_dir=SUBJECTS_DIR):
    """Load and concatenate training labels (for class weight computation)."""
    all_labels = []
    for subj_idx in train_subject_indices:
        subj_dir = os.path.join(subjects_dir, f"subject_{subj_idx:02d}")
        labels = np.load(os.path.join(subj_dir, f"labels_{target}.npy"))
        all_labels.append(labels)
    return np.concatenate(all_labels)
