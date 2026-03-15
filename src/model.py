"""
Phase 3: Full model assembly — Enhanced Att-1DCNN-GRU with TACO Cross-Attention.

Combines EEG and ECG Transformer encoders with TACO fusion and a
classification head for 3-class emotion prediction.

Supports AdamW (decoupled weight decay) with gradient clipping,
skip connections from unimodal encoders, and mixed-precision-safe softmax.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

from src import config
from src.feature_extraction import EEGTransformerEncoder, ECGTransformerEncoder
from src.taco_attention import TACOCrossAttention
from src.losses import CategoricalFocalLoss


def build_model(num_classes=None, d_model=None, num_heads=None, ff_dim=None,
                num_layers=None, dropout=None, learning_rate=None,
                lr_schedule=None, class_weights=None):
    """
    Build the complete multimodal emotion recognition model.

    Architecture:
        EEG Input (128, 9, 9) → EEGTransformerEncoder → (128, d_model)
        ECG Input (64, 64, 6)  → ECGTransformerEncoder → (64, d_model)
                                        ↓
                              TACOCrossAttention → fused (d_model)
                              + skip: GAP(EEG) ⊕ GAP(ECG) → (3*d_model)
                                        ↓
                              Dense(128) → Dropout → Dense(num_classes, softmax)

    Args:
        lr_schedule: Optional TF LearningRateSchedule (overrides learning_rate).
        class_weights: Optional dict {class_idx: weight} for focal loss alpha.
    Returns:
        Compiled Keras Model.
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if d_model is None:
        d_model = config.D_MODEL
    if num_heads is None:
        num_heads = config.NUM_HEADS
    if ff_dim is None:
        ff_dim = config.FF_DIM
    if num_layers is None:
        num_layers = config.NUM_TRANSFORMER_LAYERS
    if dropout is None:
        dropout = config.DROPOUT_RATE
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE

    # L2 only when NOT using AdamW (AdamW has decoupled weight decay)
    l2_reg = regularizers.l2(config.L2_WEIGHT_DECAY) if not config.USE_ADAMW else None

    # ── Inputs ────────────────────────────────────────────────────────────
    eeg_input = layers.Input(
        shape=(config.EEG_SEGMENT_SAMPLES, config.EEG_GRID_SIZE, config.EEG_GRID_SIZE),
        name="eeg_input",
    )
    ecg_input = layers.Input(
        shape=(config.ECG_IMAGE_SIZE, config.ECG_IMAGE_SIZE, config.ECG_IMAGE_CHANNELS),
        name="ecg_input",
    )

    # ── EEG Branch ────────────────────────────────────────────────────────
    eeg_encoder = EEGTransformerEncoder(
        d_model=d_model, num_heads=num_heads, ff_dim=ff_dim,
        num_layers=num_layers, dropout=dropout, name="eeg_encoder",
    )
    eeg_features = eeg_encoder(eeg_input)  # (batch, 128, d_model)

    # ── ECG Branch ────────────────────────────────────────────────────────
    ecg_encoder = ECGTransformerEncoder(
        d_model=d_model, num_heads=num_heads, ff_dim=ff_dim,
        num_layers=num_layers, dropout=dropout, name="ecg_encoder",
    )
    ecg_features = ecg_encoder(ecg_input)  # (batch, num_patches, d_model)

    # ── TACO Cross-Attention Fusion ───────────────────────────────────────
    taco = TACOCrossAttention(
        d_model=d_model, num_heads=num_heads, dropout=dropout,
        name="taco_fusion",
    )
    fused = taco(eeg_features, ecg_features)  # (batch, d_model)

    # ── Skip Connections: preserve unimodal features ─────────────────────
    eeg_pooled = layers.GlobalAveragePooling1D(name="eeg_skip_pool")(eeg_features)
    ecg_pooled = layers.GlobalAveragePooling1D(name="ecg_skip_pool")(ecg_features)
    combined = layers.Concatenate(name="skip_concat")([fused, eeg_pooled, ecg_pooled])

    # ── Classification Head ──────────────────────────────────────────────
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2_reg,
                     name="cls_dense1")(combined)
    x = layers.Dropout(dropout, name="cls_dropout")(x)
    # Separate logits/softmax for mixed precision stability
    logits = layers.Dense(num_classes, name="cls_output", dtype="float32")(x)
    output = layers.Activation("softmax", dtype="float32", name="cls_softmax")(logits)

    # ── Loss ──────────────────────────────────────────────────────────────
    if config.USE_FOCAL_LOSS:
        loss_fn = CategoricalFocalLoss(
            gamma=config.FOCAL_LOSS_GAMMA,
            label_smoothing=config.LABEL_SMOOTHING,
        )
        if class_weights is not None:
            loss_fn.set_alpha_from_weights(class_weights)
    elif config.LABEL_SMOOTHING > 0:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=config.LABEL_SMOOTHING,
        )
    else:
        loss_fn = "categorical_crossentropy"

    # ── Optimizer ─────────────────────────────────────────────────────────
    opt_lr = lr_schedule if lr_schedule is not None else learning_rate
    clip = config.GRADIENT_CLIP_NORM if config.GRADIENT_CLIP_NORM > 0 else None
    if config.USE_ADAMW:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=opt_lr,
            weight_decay=config.ADAMW_WEIGHT_DECAY,
            clipnorm=clip,
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=opt_lr,
            clipnorm=clip,
        )

    # ── Compile ───────────────────────────────────────────────────────────
    model = Model(inputs=[eeg_input, ecg_input], outputs=output, name="TACO_Emotion_Model")
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()

    # Quick forward pass test
    import numpy as np
    eeg_test = np.random.randn(2, 128, 9, 9).astype(np.float32)
    ecg_test = np.random.randn(2, 64, 64, 6).astype(np.float32)
    out = model.predict([eeg_test, ecg_test], verbose=0)
    print(f"\nForward pass test: EEG {eeg_test.shape}, ECG {ecg_test.shape} → output {out.shape}")
    print(f"Predictions: {out}")
