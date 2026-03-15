"""
Phase 1: Signal preprocessing — filtering, ICA artifact removal,
last-60s extraction, and 1-second segmentation.

Uses CuPy for GPU-accelerated filtering and cuML for GPU ICA when available.
Preprocessed datasets are cached to disk for instant reload.
"""

import os
import warnings

import numpy as np
from scipy.signal import iirnotch, butter, filtfilt
from sklearn.decomposition import FastICA

from src import config
from src.data_loader import load_all_subjects, get_labels
from src.utils import to_numpy, HAS_CUML

if HAS_CUML:
    from cuml.decomposition import FastICA as CuMLFastICA

# ── GPU filtfilt decision resolved once at import time ───────────────────────
try:
    from cupyx.scipy.signal import filtfilt as cu_filtfilt
    _USE_GPU_FILTFILT = True
except ImportError:
    cu_filtfilt = None
    _USE_GPU_FILTFILT = False

# ── Cache path ───────────────────────────────────────────────────────────────
CACHE_PATH = os.path.join(config.BASE_DIR, "data", "preprocessed_cache.npz")


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


def _identify_artifact_components(sources, signal=None, kurtosis_threshold=None,
                                   variance_ratio=None):
    """
    Identify ICA components likely to be artifacts using multiple heuristics.

    Heuristics:
      1. Kurtosis: Eye blinks / muscle artifacts produce super-Gaussian
         (high-kurtosis) components. Components with |kurtosis| above the
         threshold are flagged.
      2. Variance outlier: Components whose variance exceeds `variance_ratio`
         times the median variance are flagged (extreme-energy artifacts).
      3. Frontal correlation: If the original signal is provided and has ≥14
         channels (EEG), components highly correlated with frontal channels
         (AF3=0, AF4=13) are likely ocular artifacts.

    Args:
        sources: (num_samples, n_components) ICA source matrix.
        signal: Original (num_samples, num_channels) signal (optional, for
                frontal correlation check).
        kurtosis_threshold: |kurtosis| above this flags a component (default: 5.0).
        variance_ratio: Variance-to-median ratio threshold (default: 3.0).
    Returns:
        List of component indices identified as artifacts.
    """
    from scipy.stats import kurtosis as sp_kurtosis

    if kurtosis_threshold is None:
        kurtosis_threshold = config.ICA_KURTOSIS_THRESHOLD
    if variance_ratio is None:
        variance_ratio = config.ICA_VARIANCE_RATIO

    n_components = sources.shape[1]
    artifact_flags = np.zeros(n_components, dtype=bool)

    # Heuristic 1: Excess kurtosis (Fisher definition, normal = 0)
    kurt = sp_kurtosis(sources, axis=0, fisher=True)
    artifact_flags |= (np.abs(kurt) > kurtosis_threshold)

    # Heuristic 2: Variance outlier
    variances = np.var(sources, axis=0)
    median_var = np.median(variances)
    if median_var > 1e-10:
        artifact_flags |= (variances > variance_ratio * median_var)

    # Heuristic 3: High correlation with frontal EEG channels (ocular artifacts)
    if signal is not None and signal.shape[1] >= 14:
        frontal_channels = [0, 13]  # AF3, AF4
        for ch_idx in frontal_channels:
            frontal_signal = signal[:, ch_idx]
            frontal_std = np.std(frontal_signal)
            if frontal_std < 1e-8:
                continue
            for comp in range(n_components):
                comp_std = np.std(sources[:, comp])
                if comp_std < 1e-8:
                    continue
                corr = np.abs(np.corrcoef(frontal_signal, sources[:, comp])[0, 1])
                if corr > 0.8:
                    artifact_flags[comp] = True

    artifact_indices = np.where(artifact_flags)[0].tolist()

    # Safety: never remove more than half the components
    max_remove = max(1, n_components // 2)
    if len(artifact_indices) > max_remove:
        # Keep only the most suspicious ones (highest kurtosis)
        scored = sorted(artifact_indices, key=lambda c: abs(kurt[c]), reverse=True)
        artifact_indices = scored[:max_remove]

    return artifact_indices


def apply_ica(signal, n_components=None):
    """
    Apply ICA for artifact removal (eye blinks, muscle activity).

    Uses multi-heuristic artifact detection (kurtosis, variance outlier,
    frontal correlation) instead of blindly zeroing the highest-variance
    component: only components identified as likely artifacts are removed.

    Uses cuML GPU ICA when available (10-50x faster than sklearn on GPU),
    falls back to sklearn FastICA on CPU.

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

    if HAS_CUML:
        import cupy as cp
        sig_gpu = cp.asarray(signal, dtype=cp.float32)
        ica = CuMLFastICA(n_components=n_components, max_iter=200, tol=1e-3,
                          random_state=config.RANDOM_SEED)
        sources = ica.fit_transform(sig_gpu)
        sources_np = cp.asnumpy(sources)
        artifact_idx = _identify_artifact_components(sources_np, signal)
        if artifact_idx:
            sources_np[:, artifact_idx] = 0
            sources = cp.asarray(sources_np)
        # Use inverse_transform for consistent reconstruction (matches sklearn path)
        cleaned = ica.inverse_transform(sources)
        return cp.asnumpy(cleaned) if isinstance(cleaned, cp.ndarray) else np.asarray(cleaned)

    # Fallback: sklearn CPU
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning,
                                message='.*FastICA did not converge.*')
        ica = FastICA(n_components=n_components, algorithm='parallel',
                      max_iter=200, tol=1e-3,
                      random_state=config.RANDOM_SEED)
        sources = ica.fit_transform(signal)
    artifact_idx = _identify_artifact_components(sources, signal)
    if artifact_idx:
        sources[:, artifact_idx] = 0
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


# ── Cache helpers ────────────────────────────────────────────────────────────

def _save_cache(dataset, path=CACHE_PATH):
    """Save preprocessed dataset to a .npz file."""
    save_dict = {}
    for i, subj in enumerate(dataset):
        for key, arr in subj.items():
            save_dict[f"s{i}_{key}"] = arr
    save_dict["__num_subjects__"] = np.array(len(dataset))
    np.savez_compressed(path, **save_dict)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Cache saved: {path} ({size_mb:.0f} MB)")


def _load_cache(path=CACHE_PATH):
    """Load preprocessed dataset from cache."""
    data = np.load(path)
    n = int(data["__num_subjects__"])
    keys = ["eeg_segments", "ecg_segments",
            "labels_valence", "labels_arousal", "labels_dominance"]
    dataset = []
    for i in range(n):
        dataset.append({k: data[f"s{i}_{k}"] for k in keys})
    print(f"Loaded from cache: {path} ({n} subjects)")
    return dataset


def clear_cache(path=CACHE_PATH):
    """Delete the preprocessed cache file."""
    if os.path.exists(path):
        os.remove(path)
        print(f"Cache cleared: {path}")


# ── Full Dataset Builder ──────────────────────────────────────────────────────

def _preprocess_subject(subj, subj_idx, total):
    """Preprocess all trials for one subject with baseline normalization."""
    eeg_segments_all, ecg_segments_all = [], []
    labels_v, labels_a, labels_d = [], [], []

    v_scores = get_labels(subj["valence"])
    a_scores = get_labels(subj["arousal"])
    d_scores = get_labels(subj["dominance"])

    for trial_idx in range(config.NUM_TRIALS):
        eeg = preprocess_eeg(subj["eeg_stimuli"][trial_idx])
        ecg = preprocess_ecg(subj["ecg_stimuli"][trial_idx])

        # Baseline normalization: normalize stimuli relative to resting state.
        # Removes inter-subject amplitude differences — standard for DREAMER.
        eeg_base = preprocess_eeg(subj["eeg_baseline"][trial_idx])
        ecg_base = preprocess_ecg(subj["ecg_baseline"][trial_idx])

        eeg_base_mean = np.mean(eeg_base, axis=0, keepdims=True)  # (1, 14)
        eeg_base_std = np.std(eeg_base, axis=0, keepdims=True)    # (1, 14)
        eeg_base_std = np.where(eeg_base_std < 1e-8, 1.0, eeg_base_std)
        eeg = (eeg - eeg_base_mean) / eeg_base_std

        ecg_base_mean = np.mean(ecg_base, axis=0, keepdims=True)  # (1, 2)
        ecg_base_std = np.std(ecg_base, axis=0, keepdims=True)    # (1, 2)
        ecg_base_std = np.where(ecg_base_std < 1e-8, 1.0, ecg_base_std)
        ecg = (ecg - ecg_base_mean) / ecg_base_std

        eeg = extract_last_n_seconds(eeg, config.EEG_SR)
        ecg = extract_last_n_seconds(ecg, config.ECG_SR)

        eeg_segs = segment_signal(eeg, config.EEG_SR)   # (N, 128, 14)
        ecg_segs = segment_signal(ecg, config.ECG_SR)   # (N, 256, 2)

        # Per-segment z-score normalization for EEG
        # Ensures consistent scale across segments before 9x9 grid mapping
        for seg_idx in range(len(eeg_segs)):
            seg = eeg_segs[seg_idx]
            seg_mean = np.mean(seg)
            seg_std = np.std(seg)
            if seg_std > 1e-8:
                eeg_segs[seg_idx] = (seg - seg_mean) / seg_std

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


def build_dataset(mat_path=None, use_cache=True):
    """
    Full preprocessing pipeline for all subjects with disk caching.

    On the first run, preprocesses all data and saves to
    data/preprocessed_cache.npz. On subsequent runs, loads instantly
    from cache (skips all filtering/ICA).

    Args:
        mat_path: Path to DREAMER.mat (default: config.DATA_PATH).
        use_cache: If True, load from cache when available.
    Returns:
        List of 23 dicts (one per subject).
    """
    # Try loading from cache
    if use_cache and os.path.exists(CACHE_PATH):
        return _load_cache(CACHE_PATH)

    # No cache — preprocess from scratch
    all_subjects = load_all_subjects(mat_path)
    total = len(all_subjects)
    ica_backend = "cuML GPU" if HAS_CUML else "sklearn CPU"
    print(f"Preprocessing {total} subjects (ICA: {ica_backend})...")

    dataset = [
        _preprocess_subject(subj, i, total)
        for i, subj in enumerate(all_subjects)
    ]

    # Save cache for next time
    if use_cache:
        _save_cache(dataset, CACHE_PATH)

    return dataset


if __name__ == "__main__":
    dataset = build_dataset()
    total = sum(d["eeg_segments"].shape[0] for d in dataset)
    print(f"\nTotal segments across all subjects: {total}")
    print(f"Expected ~{config.NUM_SUBJECTS * config.NUM_TRIALS * config.LAST_SECONDS} "
          f"(= {config.NUM_SUBJECTS} subjects × {config.NUM_TRIALS} trials × "
          f"{config.LAST_SECONDS} segments/trial)")
