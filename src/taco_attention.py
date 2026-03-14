"""
Phase 3: TACO Cross-Attention — Token-Channel Compounded Cross Attention.

Performs simultaneous:
  - Token-wise Cross Attention (TCA): dependencies between time tokens across modalities
  - Channel-wise Cross Attention (CCA): spatial/channel correlations across modalities
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers

from src import config


class TokenCrossAttention(layers.Layer):
    """
    Token-wise Cross Attention (TCA).

    Query comes from modality A's tokens, Key/Value from modality B's tokens.
    Attention operates along the token (sequence/time) dimension.

    Input:  query (batch, seq_a, d_model), kv (batch, seq_b, d_model)
    Output: (batch, seq_a, d_model)
    """

    def __init__(self, d_model=None, num_heads=None, dropout=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model or config.D_MODEL
        self.num_heads = num_heads or config.NUM_HEADS
        self.dropout_rate = dropout or config.DROPOUT_RATE

    def build(self, input_shape):
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, query, kv, training=False):
        attn_out = self.mha(query=query, key=kv, value=kv, training=training)
        return self.norm(query + attn_out)

    def get_config(self):
        base = super().get_config()
        base.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
        return base


class ChannelCrossAttention(layers.Layer):
    """
    Channel-wise Cross Attention (CCA).

    Transposes inputs so attention operates along the channel (d_model) dimension
    instead of the token dimension. This captures correlations between
    feature channels across modalities.

    Input:  query (batch, seq_a, d_model), kv (batch, seq_b, d_model)
    Output: (batch, seq_a, d_model)
    """

    def __init__(self, d_model=None, num_heads=None, dropout=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model or config.D_MODEL
        self.num_heads = num_heads or config.NUM_HEADS
        self.dropout_rate = dropout or config.DROPOUT_RATE

    def build(self, input_shape):
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, query, kv, training=False):
        # Transpose: (batch, seq, d_model) → (batch, d_model, seq)
        query_t = tf.transpose(query, perm=[0, 2, 1])
        kv_t = tf.transpose(kv, perm=[0, 2, 1])
        # Cross attention on channel dimension
        attn_out = self.mha(query=query_t, key=kv_t, value=kv_t, training=training)
        # Transpose back: (batch, d_model, seq) → (batch, seq, d_model)
        attn_out = tf.transpose(attn_out, perm=[0, 2, 1])
        return self.norm(query + attn_out)

    def get_config(self):
        base = super().get_config()
        base.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
        return base


class TACOCrossAttention(layers.Layer):
    """
    Token-Channel Compounded (TACO) Cross Attention.

    Simultaneously applies:
      1. Token-wise Cross Attention (TCA) in both directions (EEG↔ECG)
      2. Channel-wise Cross Attention (CCA) in both directions (EEG↔ECG)

    The four outputs are concatenated and projected back to d_model.

    Input:  eeg_features (batch, seq_eeg, d_model),
            ecg_features (batch, seq_ecg, d_model)
    Output: (batch, seq_fused, d_model) — fused multimodal features
    """

    def __init__(self, d_model=None, num_heads=None, dropout=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model or config.D_MODEL
        self.num_heads = num_heads or config.NUM_HEADS
        self.dropout_rate = dropout or config.DROPOUT_RATE

    def build(self, input_shape):
        # TCA: EEG queries ECG, ECG queries EEG
        self.tca_eeg2ecg = TokenCrossAttention(
            self.d_model, self.num_heads, self.dropout_rate, name="tca_eeg2ecg"
        )
        self.tca_ecg2eeg = TokenCrossAttention(
            self.d_model, self.num_heads, self.dropout_rate, name="tca_ecg2eeg"
        )
        # CCA: EEG queries ECG, ECG queries EEG
        self.cca_eeg2ecg = ChannelCrossAttention(
            self.d_model, self.num_heads, self.dropout_rate, name="cca_eeg2ecg"
        )
        self.cca_ecg2eeg = ChannelCrossAttention(
            self.d_model, self.num_heads, self.dropout_rate, name="cca_ecg2eeg"
        )
        # Projection layers to combine cross-attention outputs (with L2)
        l2_reg = regularizers.l2(config.L2_WEIGHT_DECAY)
        self.proj_eeg = layers.Dense(self.d_model, activation="relu",
                                     kernel_regularizer=l2_reg, name="proj_eeg")
        self.proj_ecg = layers.Dense(self.d_model, activation="relu",
                                     kernel_regularizer=l2_reg, name="proj_ecg")
        self.final_norm = layers.LayerNormalization(epsilon=1e-6, name="taco_final_norm")
        self.final_proj = layers.Dense(self.d_model, kernel_regularizer=l2_reg,
                                       name="taco_final_proj")
        super().build(input_shape)

    def call(self, eeg_features, ecg_features, training=False):
        """
        Args:
            eeg_features: (batch, seq_eeg, d_model) — from EEG Transformer
            ecg_features: (batch, seq_ecg, d_model) — from ECG Transformer
        Returns:
            (batch, d_model) — fused feature vector
        """
        # Token-wise cross attention (bidirectional)
        tca_eeg = self.tca_eeg2ecg(eeg_features, ecg_features, training=training)
        tca_ecg = self.tca_ecg2eeg(ecg_features, eeg_features, training=training)

        # Channel-wise cross attention (bidirectional)
        cca_eeg = self.cca_eeg2ecg(eeg_features, ecg_features, training=training)
        cca_ecg = self.cca_ecg2eeg(ecg_features, eeg_features, training=training)

        # Combine TCA and CCA for each modality
        fused_eeg = self.proj_eeg(
            tf.concat([tca_eeg, cca_eeg], axis=-1)
        )  # (batch, seq_eeg, d_model)
        fused_ecg = self.proj_ecg(
            tf.concat([tca_ecg, cca_ecg], axis=-1)
        )  # (batch, seq_ecg, d_model)

        # Global average pooling over sequence dimension
        pooled_eeg = tf.reduce_mean(fused_eeg, axis=1)  # (batch, d_model)
        pooled_ecg = tf.reduce_mean(fused_ecg, axis=1)   # (batch, d_model)

        # Final fusion
        fused = tf.concat([pooled_eeg, pooled_ecg], axis=-1)  # (batch, 2*d_model)
        fused = self.final_proj(fused)  # (batch, d_model)
        fused = self.final_norm(fused)

        return fused

    def get_config(self):
        base = super().get_config()
        base.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
        return base
