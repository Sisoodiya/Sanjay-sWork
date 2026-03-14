"""
Phase 2: Transformer Encoder modules for EEG and ECG feature extraction.
"""

import tensorflow as tf
from tensorflow.keras import layers

from src import config
from src.spatial_encoding import SpatialPositionEncoding, TemporalPositionEncoding


class TransformerEncoderBlock(layers.Layer):
    """
    Single Transformer Encoder block: MultiHeadAttention → Add&Norm → FFN → Add&Norm.
    """

    def __init__(self, d_model=None, num_heads=None, ff_dim=None, dropout=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model or config.D_MODEL
        self.num_heads = num_heads or config.NUM_HEADS
        self.ff_dim = ff_dim or config.FF_DIM
        self.dropout_rate = dropout or config.DROPOUT_RATE

    def build(self, input_shape):
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.ff_dim, activation="relu"),
            layers.Dense(self.d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=False):
        # Self-attention
        attn_out = self.mha(x, x, x)
        attn_out = self.dropout1(attn_out, training=training)
        x = self.norm1(x + attn_out)
        # Feed-forward
        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out, training=training)
        x = self.norm2(x + ffn_out)
        return x

    def get_config(self):
        base = super().get_config()
        base.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return base


class SpatialAttentionPooling(layers.Layer):
    """
    Learnable attention-weighted pooling over spatial tokens.

    Instead of mean-pooling (which equally weights all 81 grid positions
    including the ~67 zero-padded ones), this learns to weight the 14
    electrode positions differently.

    Input:  (batch, timesteps, num_spatial_tokens, d_model)
    Output: (batch, timesteps, d_model)
    """

    def __init__(self, d_model=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model or config.D_MODEL

    def build(self, input_shape):
        self.attn_dense = layers.Dense(1, use_bias=True, name="spatial_attn_score")
        super().build(input_shape)

    def call(self, x):
        # x: (batch, timesteps, num_tokens, d_model)
        scores = self.attn_dense(x)  # (batch, timesteps, num_tokens, 1)
        weights = tf.nn.softmax(scores, axis=2)  # Softmax over spatial dim
        pooled = tf.reduce_sum(x * weights, axis=2)  # (batch, timesteps, d_model)
        return pooled

    def get_config(self):
        base = super().get_config()
        base.update({"d_model": self.d_model})
        return base


class EEGTransformerEncoder(layers.Layer):
    """
    EEG feature extractor with deferred spatial pooling:
      (batch, 128, 9, 9) → SpatialPositionEncoding → (batch, 128, 81, d_model)
      → Per-timestep Transformer over spatial tokens →
      → Learnable spatial attention pooling → (batch, 128, d_model)
      → Temporal positional encoding → Transformer stack → (batch, 128, d_model)

    Key improvement: spatial tokens are processed by a Transformer BEFORE
    being pooled. This lets the model learn spatial relationships between
    electrode positions before compressing to a per-timestep representation.
    """

    def __init__(self, d_model=None, num_heads=None, ff_dim=None,
                 num_layers=None, dropout=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model or config.D_MODEL
        self.num_heads = num_heads or config.NUM_HEADS
        self.ff_dim = ff_dim or config.FF_DIM
        self.num_layers = num_layers or config.NUM_TRANSFORMER_LAYERS
        self.dropout_rate = dropout or config.DROPOUT_RATE

    def build(self, input_shape):
        self.spatial_encoding = SpatialPositionEncoding(
            d_model=self.d_model, name="eeg_spatial_enc"
        )
        # Spatial Transformer: processes 81 spatial tokens per timestep
        self.spatial_transformer = TransformerEncoderBlock(
            d_model=self.d_model, num_heads=self.num_heads,
            ff_dim=self.ff_dim, dropout=self.dropout_rate,
            name="eeg_spatial_transformer"
        )
        # Learnable spatial attention pooling (replaces mean pooling)
        self.spatial_pool = SpatialAttentionPooling(
            d_model=self.d_model, name="eeg_spatial_pool"
        )
        self.temporal_pos = TemporalPositionEncoding(
            max_len=256, d_model=self.d_model, name="eeg_temporal_pos"
        )
        # Temporal Transformer stack
        self.transformer_blocks = [
            TransformerEncoderBlock(
                d_model=self.d_model, num_heads=self.num_heads,
                ff_dim=self.ff_dim, dropout=self.dropout_rate,
                name=f"eeg_transformer_{i}"
            )
            for i in range(self.num_layers)
        ]
        super().build(input_shape)

    def call(self, x, training=False):
        """
        Args:
            x: (batch, 128, 9, 9)
        Returns:
            (batch, 128, d_model)
        """
        batch_size = tf.shape(x)[0]
        timesteps = tf.shape(x)[1]

        # Spatial encoding: (batch, 128, 81, d_model)
        x = self.spatial_encoding(x)

        # Process spatial tokens with Transformer (per timestep):
        # Reshape to (batch * timesteps, 81, d_model) for efficient batching
        num_tokens = x.shape[2] or tf.shape(x)[2]  # 81
        x = tf.reshape(x, [batch_size * timesteps, num_tokens, self.d_model])
        x = self.spatial_transformer(x, training=training)
        # Reshape back: (batch, timesteps, 81, d_model)
        x = tf.reshape(x, [batch_size, timesteps, num_tokens, self.d_model])

        # Learnable attention pooling over spatial tokens: (batch, 128, d_model)
        x = self.spatial_pool(x)

        # Add temporal positional encoding
        x = self.temporal_pos(x)
        # Temporal Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return x

    def get_config(self):
        base = super().get_config()
        base.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "ff_dim": self.ff_dim, "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
        })
        return base


class ECGPatchEmbedding(layers.Layer):
    """
    Convert ECG 2D image into patch embeddings.

    Input:  (batch, image_size, image_size, 6)
    Output: (batch, num_patches, d_model)

    Splits the image into non-overlapping patches, then linearly projects each.
    """

    def __init__(self, patch_size=None, d_model=None, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size or config.ECG_PATCH_SIZE
        self.d_model = d_model or config.D_MODEL

    def build(self, input_shape):
        image_size = input_shape[1]
        num_channels = input_shape[3]
        self.num_patches = (image_size // self.patch_size) ** 2
        patch_dim = self.patch_size * self.patch_size * num_channels
        self.projection = layers.Dense(self.d_model)
        self.patch_dim = patch_dim
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        # Extract patches using tf.image.extract_patches
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # Reshape: (batch, num_patches_h, num_patches_w, patch_dim) → (batch, num_patches, patch_dim)
        patches = tf.reshape(patches, [batch_size, self.num_patches, self.patch_dim])
        # Project to d_model
        return self.projection(patches)

    def get_config(self):
        base = super().get_config()
        base.update({"patch_size": self.patch_size, "d_model": self.d_model})
        return base


class ECGTransformerEncoder(layers.Layer):
    """
    ECG feature extractor:
      (batch, 64, 64, 6) → PatchEmbedding → PositionalEncoding →
      TransformerEncoder stack → (batch, num_patches, d_model)
    """

    def __init__(self, patch_size=None, d_model=None, num_heads=None,
                 ff_dim=None, num_layers=None, dropout=None, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size or config.ECG_PATCH_SIZE
        self.d_model = d_model or config.D_MODEL
        self.num_heads = num_heads or config.NUM_HEADS
        self.ff_dim = ff_dim or config.FF_DIM
        self.num_layers = num_layers or config.NUM_TRANSFORMER_LAYERS
        self.dropout_rate = dropout or config.DROPOUT_RATE

    def build(self, input_shape):
        self.patch_embedding = ECGPatchEmbedding(
            patch_size=self.patch_size, d_model=self.d_model,
            name="ecg_patch_embed"
        )
        self.temporal_pos = TemporalPositionEncoding(
            max_len=256, d_model=self.d_model, name="ecg_temporal_pos"
        )
        self.transformer_blocks = [
            TransformerEncoderBlock(
                d_model=self.d_model, num_heads=self.num_heads,
                ff_dim=self.ff_dim, dropout=self.dropout_rate,
                name=f"ecg_transformer_{i}"
            )
            for i in range(self.num_layers)
        ]
        super().build(input_shape)

    def call(self, x, training=False):
        """
        Args:
            x: (batch, 64, 64, 6)
        Returns:
            (batch, num_patches, d_model)
        """
        # Patch embedding: (batch, 64, d_model)
        x = self.patch_embedding(x)
        # Positional encoding
        x = self.temporal_pos(x)
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return x

    def get_config(self):
        base = super().get_config()
        base.update({
            "patch_size": self.patch_size, "d_model": self.d_model,
            "num_heads": self.num_heads, "ff_dim": self.ff_dim,
            "num_layers": self.num_layers, "dropout": self.dropout_rate,
        })
        return base
