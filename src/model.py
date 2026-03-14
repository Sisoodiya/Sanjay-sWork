"""
Phase 3: Full model assembly — Enhanced Att-1DCNN-GRU with TACO Cross-Attention.

Combines EEG and ECG Transformer encoders with TACO fusion and a
classification head for 3-class emotion prediction.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from src import config
from src.feature_extraction import EEGTransformerEncoder, ECGTransformerEncoder
from src.taco_attention import TACOCrossAttention


def build_model(num_classes=None, d_model=None, num_heads=None, ff_dim=None,
                num_layers=None, dropout=None, learning_rate=None):
    """
    Build the complete multimodal emotion recognition model.

    Architecture:
        EEG Input (128, 9, 9) → EEGTransformerEncoder → (128, d_model)
        ECG Input (64, 64, 6)  → ECGTransformerEncoder → (64, d_model)
                                        ↓
                              TACOCrossAttention
                                        ↓
                              Dense(128) → Dropout → Dense(num_classes, softmax)

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

    # ── Classification Head ───────────────────────────────────────────────
    x = layers.Dense(128, activation="relu", name="cls_dense1")(fused)
    x = layers.Dropout(dropout, name="cls_dropout")(x)
    output = layers.Dense(num_classes, activation="softmax", name="cls_output")(x)

    # ── Compile ───────────────────────────────────────────────────────────
    model = Model(inputs=[eeg_input, ecg_input], outputs=output, name="TACO_Emotion_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
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
