"""
Entry point: LOSOCV Training Loop — Leave-One-Subject-Out Cross-Validation.

Run from project root:
    python train.py --target valence
    python train.py --target all
    python train.py --target arousal --subjects 2  (quick test)
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from src import config
from src.utils import set_seed, get_class_weights, ensure_dir
from src.preprocessing import build_dataset
from src.transforms import transform_ecg_batch, transform_eeg_batch
from src.augmentation import augment_training_data
from src.model import build_model
from src.evaluate import (
    compute_metrics, print_classification_report,
    plot_confusion_matrix, print_results_table,
)


def prepare_fold_data(dataset, test_subject_idx, target):
    """
    Split dataset into train, validation, and test for one LOSOCV fold.

    Uses a separate training subject as validation to avoid data leakage.
    The validation subject is chosen as the one immediately before the test
    subject (wrapping around), ensuring the test subject is never used for
    EarlyStopping or ReduceLROnPlateau decisions.

    Args:
        dataset: List of 23 subject dicts from build_dataset().
        test_subject_idx: Index of the held-out test subject.
        target: 'valence', 'arousal', or 'dominance'.
    Returns:
        (train_eeg, train_ecg, train_labels,
         val_eeg, val_ecg, val_labels,
         test_eeg, test_ecg, test_labels)
    """
    label_key = f"labels_{target}"
    n_subjects = len(dataset)

    # Pick a validation subject: the one before the test subject (wrapping)
    val_subject_idx = (test_subject_idx - 1) % n_subjects

    train_eeg, train_ecg, train_labels = [], [], []
    for i, subj in enumerate(dataset):
        if i == test_subject_idx or i == val_subject_idx:
            continue
        train_eeg.append(subj["eeg_segments"])
        train_ecg.append(subj["ecg_segments"])
        train_labels.append(subj[label_key])

    train_eeg = np.concatenate(train_eeg, axis=0)
    train_ecg = np.concatenate(train_ecg, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    val_eeg = dataset[val_subject_idx]["eeg_segments"]
    val_ecg = dataset[val_subject_idx]["ecg_segments"]
    val_labels = dataset[val_subject_idx][label_key]

    test_eeg = dataset[test_subject_idx]["eeg_segments"]
    test_ecg = dataset[test_subject_idx]["ecg_segments"]
    test_labels = dataset[test_subject_idx][label_key]

    return (train_eeg, train_ecg, train_labels,
            val_eeg, val_ecg, val_labels,
            test_eeg, test_ecg, test_labels)


def apply_transforms(eeg_segments, ecg_segments):
    """Apply 2D transformations to raw segments."""
    print("  Applying EEG 2D grid transform...")
    eeg_2d = transform_eeg_batch(eeg_segments)
    print("  Applying ECG 2D transforms (GAF+RP+MTF)...")
    ecg_2d = transform_ecg_batch(ecg_segments)
    return eeg_2d, ecg_2d


def losocv_train(dataset, target="valence", num_subjects=None, save_dir=None):
    """
    Run Leave-One-Subject-Out Cross-Validation.

    Args:
        dataset: List of subject dicts from build_dataset().
        target: 'valence', 'arousal', or 'dominance'.
        num_subjects: Number of subjects to evaluate (for testing; default: all 23).
        save_dir: Directory to save results.
    Returns:
        List of per-subject result dicts.
    """
    if num_subjects is None:
        num_subjects = len(dataset)
    if save_dir is None:
        save_dir = os.path.join(config.SAVE_DIR, target)
    ensure_dir(save_dir)

    all_y_true = []
    all_y_pred = []
    subject_results = []

    for fold in range(num_subjects):
        print(f"\n{'=' * 60}")
        print(f"LOSOCV Fold {fold + 1}/{num_subjects} — Test Subject: {fold}")
        print(f"{'=' * 60}")

        # Split data (train / validation / test)
        (train_eeg, train_ecg, train_labels,
         val_eeg, val_ecg, val_labels,
         test_eeg, test_ecg, test_labels) = prepare_fold_data(dataset, fold, target)

        val_subject_idx = (fold - 1) % len(dataset)
        print(f"  Train: {len(train_labels)} segments (21 subjects), "
              f"Val: {len(val_labels)} segments (subject {val_subject_idx}), "
              f"Test: {len(test_labels)} segments (subject {fold})")

        # Apply 2D transforms
        print("  Transforming training data...")
        train_eeg_2d, train_ecg_2d = apply_transforms(train_eeg, train_ecg)
        print("  Transforming validation data...")
        val_eeg_2d, val_ecg_2d = apply_transforms(val_eeg, val_ecg)
        print("  Transforming test data...")
        test_eeg_2d, test_ecg_2d = apply_transforms(test_eeg, test_ecg)

        # One-hot encode labels
        train_labels_oh = to_categorical(train_labels, config.NUM_CLASSES)
        val_labels_oh = to_categorical(val_labels, config.NUM_CLASSES)

        # Data augmentation (training data only)
        n_before = len(train_labels_oh)
        train_eeg_2d, train_ecg_2d, train_labels_oh = augment_training_data(
            train_eeg_2d, train_ecg_2d, train_labels_oh,
        )
        print(f"  Augmentation: {n_before} → {len(train_labels_oh)} training segments")
        test_labels_oh = to_categorical(test_labels, config.NUM_CLASSES)

        # Class weights
        class_weights = get_class_weights(train_labels)
        print(f"  Class weights: {class_weights}")

        # Build LR schedule (warmup + cosine decay)
        steps_per_epoch = len(train_labels_oh) // config.BATCH_SIZE + 1
        if config.LR_USE_COSINE_DECAY:
            from src.lr_schedule import build_lr_schedule
            lr_schedule = build_lr_schedule(steps_per_epoch)
            print(f"  LR schedule: warmup {config.LR_WARMUP_EPOCHS} epochs → cosine decay")
        else:
            lr_schedule = None

        # Build model (fresh for each fold; passes class weights for focal loss)
        model = build_model(lr_schedule=lr_schedule, class_weights=class_weights)

        # Callbacks (monitor validation loss — from a separate subject, not test)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            ),
        ]
        # Only add ReduceLROnPlateau when NOT using cosine decay schedule
        if not config.LR_USE_COSINE_DECAY:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
            ))

        # When using focal loss, class weights are handled by the loss's alpha
        # parameter — do NOT also pass class_weight to model.fit (double-counting)
        fit_class_weight = None if config.USE_FOCAL_LOSS else class_weights

        # Train (validation data is a separate subject, NOT the test subject)
        model.fit(
            [train_eeg_2d, train_ecg_2d],
            train_labels_oh,
            validation_data=([val_eeg_2d, val_ecg_2d], val_labels_oh),
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            class_weight=fit_class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        # Predict
        pred_probs = model.predict([test_eeg_2d, test_ecg_2d], verbose=0)
        y_pred = np.argmax(pred_probs, axis=1)
        y_true = test_labels

        # Metrics
        metrics = compute_metrics(y_true, y_pred)
        subject_results.append(metrics)
        print(f"  Subject {fold} — Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        # Collect for overall evaluation
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        # Save confusion matrix plot
        plot_confusion_matrix(
            y_true, y_pred,
            target_name=f"{target} — Subject {fold}",
            save_path=os.path.join(save_dir, f"cm_subject_{fold}.png"),
        )

        # Save model weights for this fold
        model.save_weights(os.path.join(save_dir, f"model_fold_{fold}.weights.h5"))

        # Clear session to free memory
        tf.keras.backend.clear_session()

    # Overall results
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    print_classification_report(all_y_true, all_y_pred, target_name=target.upper())
    plot_confusion_matrix(
        all_y_true, all_y_pred,
        target_name=f"{target} — Overall",
        save_path=os.path.join(save_dir, "cm_overall.png"),
    )

    return subject_results


def main():
    parser = argparse.ArgumentParser(description="Train TACO Emotion Recognition Model (LOSOCV)")
    parser.add_argument("--target", type=str, default="valence",
                        choices=["valence", "arousal", "dominance", "all"],
                        help="Target emotion dimension")
    parser.add_argument("--subjects", type=int, default=None,
                        help="Number of subjects to evaluate (for quick testing)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to DREAMER.mat")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)

    # Build preprocessed dataset
    print("Building preprocessed dataset...")
    dataset = build_dataset(args.data)

    targets = config.TARGETS if args.target == "all" else [args.target]
    all_results = {}

    for target in targets:
        print(f"\n{'#' * 70}")
        print(f"  TRAINING: {target.upper()}")
        print(f"{'#' * 70}")
        results = losocv_train(dataset, target=target, num_subjects=args.subjects)
        all_results[target] = results

    # Print summary table
    print_results_table(all_results)


if __name__ == "__main__":
    main()
