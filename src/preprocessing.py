"""
Phase 1: Signal preprocessing — filtering, ICA artifact removal,
last-60s extraction, and 1-second segmentation.

Uses CuPy for GPU-accelerated filtering when available.
Subjects are preprocessed in parallel via joblib.
"""

import warnings

import numpy as np
from scipy.signal import iirnotch, butter, filtfilt
from sklearn.decomposition import FastICA
from joblib import Parallel, delayed

from src import config
from src.data_loader import load_all_subjects, get_labels
from src.utils import to_numpy

# ── GPU filtfilt decision resolved once at import time ───────────────────────
try:
    from cupyx.scipy.signal import filtfilt as cu_filtfilt
    _USE_GPU_FILTFILT = True
except ImportError:
    cu_filtfilt = None
    _USE_GPU_FILTFILT = False


def _gpu_filtfilt(b, a, signal):
    """Run filtfilt on GPU if CuPy is available, else CPU."""
    if _USE_GPU_FILTFILT:
        import cupy as cp
        result = cu_filtfilt(cp.asarray(b), cp.asarray(a), cp.asarray(signal), axis=0)
        return to_numpy(result)
    return filtfilt(b, a, signal, axis=0)


# ── Filters ───────────────────────────────────────────────────────────────────

def notch_filter(signal, fs, freq=50.0, quality=30.0):
    """
    Apply a notch filter to remove power-line interference.

    Args:
        signal: (num_samples, num_channels) array.
        fs: Sampling frequency.
        freq: Notch frequency (default 50 Hz).
        quality: Quality factor.
    Returns:
        Filtered signal, same shape.
    """
    b, a = iirnotch(freq, quality, fs)
    return _gpu_filtfilt(b, a, signal)


def bandpass_filter(signal, fs, low=0.5, high=45.0, order=4):
    """
    Apply a 4th-order Butterworth bandpass filter.

    Args:
        signal: (num_samples, num_channels) array.
        fs: Sampling frequency.
        low: Lower cutoff frequency.
        high: Upper cutoff frequency.
        order: Filter order.
    Returns:
        Filtered signal, same shape.
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return _gpu_filtfilt(b, a, signal)


def apply_ica(signal, n_components=None):
    """
    Apply ICA for artifact removal (eye blinks, muscle activity).

    Uses the parallel algorithm (3-5x faster than deflation).
    Partial convergence is acceptable — the dominant artifact component
    is still identified.

    Args:
        signal: (num_samples, num_channels) array.
        n_components: Number of ICA components. Defaults to num_channels.
    Returns:
        Cleaned signal, same shape.
    """
    if n_components is None:
        n_components = signal.shape[1]
    if np.std(signal) < 1e-8:
        return signal
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning,
                                message='.*FastICA did not converge.*')
        ica = FastICA(n_components=n_components, algorithm='parallel',
                      max_iter=200, tol=1e-3,
                      random_state=config.RANDOM_SEED)
        sources = ica.fit_transform(signal)
    variances = np.var(sources, axis=0)
    sources[:, np.argmax(variances)] = 0
    return ica.inverse_transform(sources)


# ── Preprocessing Pipelines ──────────────────────────────────────────────────

def preprocess_eeg(eeg_signal, fs=None):
    """
    Full EEG preprocessing: notch → bandpass → ICA.

    Args:
        eeg_signal: (num_samples, 14) array.
        fs: Sampling rate (default: config.EEG_SR).
    Returns:
        Preprocessed signal, same shape.
    """
    if fs is None:
        fs = config.EEG_SR
    signal = notch_filter(eeg_signal, fs)
    signal = bandpass_filter(signal, fs, low=0.5, high=45.0)
    signal = apply_ica(signal, n_components=config.EEG_CHANNELS)
    return signal


def preprocess_ecg(ecg_signal, fs=None):
    """
    ECG preprocessing: notch → bandpass.
    (ICA not applied to 2-channel ECG.)

    Args:
        ecg_signal: (num_samples, 2) array.
        fs: Sampling rate (default: config.ECG_SR).
    Returns:
        Preprocessed signal, same shape.
    """
    if fs is None:
        fs = config.ECG_SR
    signal = notch_filter(ecg_signal, fs)
    signal = bandpass_filter(signal, fs, low=0.5, high=45.0)
    return signal


# ── Extraction & Segmentation ────────────────────────────────────────────────

def extract_last_n_seconds(signal, fs, n_seconds=None):
    """
    Extract the last N seconds from a signal.

    Args:
        signal: (num_samples, num_channels) array.
        fs: Sampling frequency.
        n_seconds: Number of seconds to extract (default: config.LAST_SECONDS).
    Returns:
        (n_seconds * fs, num_channels) array. If signal is shorter, returns as-is.
    """
    if n_seconds is None:
        n_seconds = config.LAST_SECONDS
    n_samples = int(n_seconds * fs)
    if signal.shape[0] <= n_samples:
        return signal
    return signal[-n_samples:]


def segment_signal(signal, fs, segment_sec=None):
    """
    Divide signal into non-overlapping fixed-size segments.

    Args:
        signal: (num_samples, num_channels) array.
        fs: Sampling frequency.
        segment_sec: Segment length in seconds (default: config.SEGMENT_LENGTH_SEC).
    Returns:
        (num_segments, segment_samples, num_channels) array.
    """
    if segment_sec is None:
        segment_sec = config.SEGMENT_LENGTH_SEC
    segment_samples = int(segment_sec * fs)
    n_segments = signal.shape[0] // segment_samples
    trimmed = signal[: n_segments * segment_samples]
    return trimmed.reshape(n_segments, segment_samples, -1)


# ── Full Dataset Builder ──────────────────────────────────────────────────────

def _preprocess_subject(subj, subj_idx, total):
    """Preprocess all trials for one subject. Called in parallel by build_dataset."""
    eeg_segments_all, ecg_segments_all = [], []
    labels_v, labels_a, labels_d = [], [], []

    v_scores = get_labels(subj["valence"])
    a_scores = get_labels(subj["arousal"])
    d_scores = get_labels(subj["dominance"])

    for trial_idx in range(config.NUM_TRIALS):
        eeg = preprocess_eeg(subj["eeg_stimuli"][trial_idx])
        ecg = preprocess_ecg(subj["ecg_stimuli"][trial_idx])

        eeg = extract_last_n_seconds(eeg, config.EEG_SR)
        ecg = extract_last_n_seconds(ecg, config.ECG_SR)

        eeg_segs = segment_signal(eeg, config.EEG_SR)   # (N, 128, 14)
        ecg_segs = segment_signal(ecg, config.ECG_SR)   # (N, 256, 2)

        n_segs = min(len(eeg_segs), len(ecg_segs))
        eeg_segments_all.append(eeg_segs[:n_segs])
        ecg_segments_all.append(ecg_segs[:n_segs])

        labels_v.extend([v_scores[trial_idx]] * n_segs)
        labels_a.extend([a_scores[trial_idx]] * n_segs)
        labels_d.extend([d_scores[trial_idx]] * n_segs)

    eeg_arr = np.concatenate(eeg_segments_all, axis=0)
    ecg_arr = np.concatenate(ecg_segments_all, axis=0)
    print(f"  Subject {subj_idx + 1}/{total}: "
          f"EEG {eeg_arr.shape}, ECG {ecg_arr.shape}, "
          f"Labels {len(labels_v)}")
    return {
        "eeg_segments": eeg_arr,
        "ecg_segments": ecg_arr,
        "labels_valence": np.array(labels_v),
        "labels_arousal": np.array(labels_a),
        "labels_dominance": np.array(labels_d),
    }


def build_dataset(mat_path=None):
    """
    Full preprocessing pipeline for all subjects, run in parallel.

    For each subject and trial:
      1. Preprocess EEG and ECG signals
      2. Extract last 60 seconds
      3. Segment into 1-second windows

    Returns:
        List of 23 dicts (one per subject), each containing:
            'eeg_segments': (num_total_segments, 128, 14) array
            'ecg_segments': (num_total_segments, 256, 2) array
            'labels_valence': (num_total_segments,) array of class labels
            'labels_arousal': (num_total_segments,) array of class labels
            'labels_dominance': (num_total_segments,) array of class labels
    """
    all_subjects = load_all_subjects(mat_path)
    total = len(all_subjects)
    print(f"Preprocessing {total} subjects in parallel...")

    dataset = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_preprocess_subject)(subj, i, total)
        for i, subj in enumerate(all_subjects)
    )
    return dataset


if __name__ == "__main__":
    dataset = build_dataset()
    total = sum(d["eeg_segments"].shape[0] for d in dataset)
    print(f"\nTotal segments across all subjects: {total}")
    print(f"Expected ~{config.NUM_SUBJECTS * config.NUM_TRIALS * config.LAST_SECONDS} "
          f"(= {config.NUM_SUBJECTS} subjects × {config.NUM_TRIALS} trials × "
          f"{config.LAST_SECONDS} segments/trial)")
