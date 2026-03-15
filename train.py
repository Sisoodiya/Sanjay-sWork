"""
Entry point: LOSOCV Training Loop — Leave-One-Subject-Out Cross-Validation.

Run from project root:
    python train.py --target valence
    python train.py --target all
    python train.py --target arousal --subjects 2  (quick test)
"""

import os
import gc
import argparse
import numpy as np
import tensorflow as tf

from src import config
from src.utils import set_seed, get_class_weights, ensure_dir
from src.model import build_model
from src.evaluate import (
    compute_metrics, print_classification_report,
    plot_confusion_matrix, print_results_table,
)
from src.data_pipeline import (
    save_subject_files, subjects_cache_exists,
    make_training_dataset, load_eval_data,
    count_training_samples, get_training_labels,
)


def losocv_train(target="valence", num_subjects=None, save_dir=None):
    """
    Run Leave-One-Subject-Out Cross-Validation using tf.data streaming.

    Data is loaded from per-subject .npy files on disk. At most one
    subject's 2D data (~76 MB) is in RAM at a time during training.

    Args:
        target: 'valence', 'arousal', or 'dominance'.
        num_subjects: Number of folds to run (default: all 23).
        save_dir: Directory to save results.
    Returns:
        List of per-subject result dicts.
    """
    n_total = num_subjects if num_subjects else config.NUM_SUBJECTS
    if save_dir is None:
        save_dir = os.path.join(config.SAVE_DIR, target)
    ensure_dir(save_dir)

    all_y_true = []
    all_y_pred = []
    subject_results = []

    for fold in range(n_total):
        print(f"\n{'=' * 60}")
        print(f"LOSOCV Fold {fold + 1}/{n_total} — Test Subject: {fold}")
        print(f"{'=' * 60}")

        val_subject_idx = (fold - 1) % config.NUM_SUBJECTS
        train_indices = [i for i in range(config.NUM_SUBJECTS)
                         if i != fold and i != val_subject_idx]

        # Build tf.data training pipeline (streams from disk)
        train_ds = make_training_dataset(train_indices, target)

        # Load val/test as numpy (one subject each, ~76 MB)
        val_eeg, val_ecg, val_labels, val_labels_oh = load_eval_data(
            val_subject_idx, target)
        test_eeg, test_ecg, test_labels, test_labels_oh = load_eval_data(
            fold, target)

        # Class weights (loads only small label arrays)
        train_labels_flat = get_training_labels(train_indices, target)
        class_weights = get_class_weights(train_labels_flat)
        del train_labels_flat

        # Steps per epoch for LR schedule
        n_train_total = count_training_samples(train_indices, target)
        steps_per_epoch = n_train_total // config.BATCH_SIZE

        print(f"  Train: ~{n_train_total} samples (21 subjects + aug), "
              f"Val: {len(val_labels)} (subject {val_subject_idx}), "
              f"Test: {len(test_labels)} (subject {fold})")
        print(f"  Class weights: {class_weights}")

        # Build LR schedule (warmup + cosine decay)
        if config.LR_USE_COSINE_DECAY:
            from src.lr_schedule import build_lr_schedule
            lr_schedule = build_lr_schedule(steps_per_epoch)
            print(f"  LR schedule: warmup {config.LR_WARMUP_EPOCHS} epochs → cosine decay")
        else:
            lr_schedule = None

        # Build model (fresh for each fold)
        model = build_model(lr_schedule=lr_schedule, class_weights=class_weights)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            ),
        ]
        if not config.LR_USE_COSINE_DECAY:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
            ))

        fit_class_weight = None if config.USE_FOCAL_LOSS else class_weights

        # Train with tf.data pipeline (no numpy copy overhead)
        model.fit(
            train_ds,
            validation_data=([val_eeg, val_ecg], val_labels_oh),
            epochs=config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            class_weight=fit_class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        # Predict
        pred_probs = model.predict([test_eeg, test_ecg], verbose=0)
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

        # Clear session and free memory
        del train_ds, val_eeg, val_ecg, val_labels, val_labels_oh
        del test_eeg, test_ecg, test_labels, test_labels_oh
        del model
        tf.keras.backend.clear_session()
        gc.collect()

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

    # Allow TF and CuPy to share GPU memory without conflicts
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Enable Mixed Precision to handle float16 inputs efficiently
    # and utilize the T4 GPU's Tensor Cores for massive speedups.
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"GPU: {[g.name for g in gpus] if gpus else 'None (CPU only)'}")
    print("Enabled Mixed Precision (float16) for GPU Tensor Cores.")

    set_seed(config.RANDOM_SEED)

    # Ensure per-subject 2D cache exists on disk
    if not subjects_cache_exists():
        print("Building per-subject 2D cache (one-time)...")
        from src.preprocessing import build_dataset
        dataset = build_dataset(args.data)
        save_subject_files(dataset)
        del dataset
        gc.collect()
        print("Per-subject 2D cache ready.\n")
    else:
        print("Per-subject 2D cache found. Skipping preprocessing.\n")

    # Run LOSOCV — data streams from disk, no dataset object in memory
    targets = config.TARGETS if args.target == "all" else [args.target]
    all_results = {}

    for target in targets:
        print(f"\n{'#' * 70}")
        print(f"  TRAINING: {target.upper()}")
        print(f"{'#' * 70}")
        results = losocv_train(target=target, num_subjects=args.subjects)
        all_results[target] = results

    # Print summary table
    print_results_table(all_results)


if __name__ == "__main__":
    main()
