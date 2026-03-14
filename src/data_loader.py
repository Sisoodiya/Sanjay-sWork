"""
Phase 1: Load DREAMER.mat and extract per-subject EEG/ECG signals and VAD labels.
"""

import numpy as np
import scipy.io as sio
from src import config


def load_dreamer(mat_path=None):
    """
    Load the DREAMER.mat file.
    Returns the DREAMER struct (numpy structured array).
    """
    if mat_path is None:
        mat_path = config.DATA_PATH
    data = sio.loadmat(mat_path, squeeze_me=False)
    dreamer = data["DREAMER"][0, 0]
    return dreamer


def extract_subject_data(dreamer, subject_idx):
    """
    Extract all trial data for a single subject.

    Args:
        dreamer: The loaded DREAMER struct.
        subject_idx: Subject index (0-22).

    Returns:
        dict with keys:
            'eeg_stimuli': list of 18 arrays, each (num_samples, 14)
            'ecg_stimuli': list of 18 arrays, each (num_samples, 2)
            'eeg_baseline': list of 18 arrays, each (7808, 14)
            'ecg_baseline': list of 18 arrays, each (15616, 2)
            'valence': (18,) array of scores 1-5
            'arousal': (18,) array of scores 1-5
            'dominance': (18,) array of scores 1-5
            'age': str
            'gender': str
    """
    subject = dreamer["Data"][0, subject_idx]

    eeg_stimuli = []
    ecg_stimuli = []
    eeg_baseline = []
    ecg_baseline = []

    for trial_idx in range(config.NUM_TRIALS):
        eeg_stim = subject["EEG"][0, 0]["stimuli"][0, 0][trial_idx, 0]
        ecg_stim = subject["ECG"][0, 0]["stimuli"][0, 0][trial_idx, 0]
        eeg_base = subject["EEG"][0, 0]["baseline"][0, 0][trial_idx, 0]
        ecg_base = subject["ECG"][0, 0]["baseline"][0, 0][trial_idx, 0]

        eeg_stimuli.append(eeg_stim.astype(np.float32))
        ecg_stimuli.append(ecg_stim.astype(np.float32))
        eeg_baseline.append(eeg_base.astype(np.float32))
        ecg_baseline.append(ecg_base.astype(np.float32))

    valence = subject["ScoreValence"][0, 0].flatten().astype(int)
    arousal = subject["ScoreArousal"][0, 0].flatten().astype(int)
    dominance = subject["ScoreDominance"][0, 0].flatten().astype(int)

    age = str(subject["Age"][0, 0].flat[0])
    gender = str(subject["Gender"][0, 0].flat[0])

    return {
        "eeg_stimuli": eeg_stimuli,
        "ecg_stimuli": ecg_stimuli,
        "eeg_baseline": eeg_baseline,
        "ecg_baseline": ecg_baseline,
        "valence": valence,
        "arousal": arousal,
        "dominance": dominance,
        "age": age,
        "gender": gender,
    }


def get_labels(scores, label_map=None):
    """
    Convert 1-5 scores to 3-class labels.

    Args:
        scores: array of scores (1-5).
        label_map: dict {score: class}. Defaults to config.LABEL_MAP.

    Returns:
        array of class labels (0, 1, 2).
    """
    if label_map is None:
        label_map = config.LABEL_MAP
    return np.array([label_map[s] for s in scores])


def load_all_subjects(mat_path=None):
    """
    Load all 23 subjects' data.

    Returns:
        list of 23 dicts (one per subject), each from extract_subject_data().
    """
    dreamer = load_dreamer(mat_path)
    all_subjects = []
    for s in range(config.NUM_SUBJECTS):
        subject_data = extract_subject_data(dreamer, s)
        all_subjects.append(subject_data)
    return all_subjects


if __name__ == "__main__":
    print("Loading DREAMER dataset...")
    subjects = load_all_subjects()
    print(f"Loaded {len(subjects)} subjects")
    for i, s in enumerate(subjects):
        print(
            f"  Subject {i}: age={s['age']}, gender={s['gender']}, "
            f"EEG trial 0 shape={s['eeg_stimuli'][0].shape}, "
            f"ECG trial 0 shape={s['ecg_stimuli'][0].shape}, "
            f"Valence={s['valence']}"
        )
