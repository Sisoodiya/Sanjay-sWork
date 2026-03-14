"""
Phase 2: 2D Spatial Position Encoding for EEG grid representations.

Uses sinusoidal encoding to preserve the physical spatial relationships
between electrode positions on the scalp.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src import config


def create_2d_positional_encoding(grid_h, grid_w, d_model):
    """
    Generate 2D sinusoidal positional encoding.

    Vertical positions use sine, horizontal positions use cosine.

    Args:
        grid_h: Grid height.
        grid_w: Grid width.
        d_model: Embedding dimension.
    Returns:
        (grid_h * grid_w, d_model) numpy array.
    """
    pe = np.zeros((grid_h, grid_w, d_model), dtype=np.float32)
    d_half = d_model // 2

    for row in range(grid_h):
        for col in range(grid_w):
            for i in range(d_half):
                denom = 10000 ** (2 * i / d_model)
                pe[row, col, 2 * i] = np.sin(row / denom)
                pe[row, col, 2 * i + 1] = np.cos(col / denom)

    # Flatten grid: (grid_h * grid_w, d_model)
    return pe.reshape(grid_h * grid_w, d_model)


class SpatialPositionEncoding(layers.Layer):
    """
    Keras layer that applies 2D spatial position encoding to EEG grid data.

    Input:  (batch, timesteps, grid_h, grid_w)
    Output: (batch, timesteps, grid_h * grid_w, d_model)

    For each timestep, the 9×9 grid is flattened to 81 tokens, projected to
    d_model dimensions, and 2D positional encoding is added.
    """

    def __init__(self, grid_h=None, grid_w=None, d_model=None, **kwargs):
        super().__init__(**kwargs)
        self.grid_h = grid_h or config.EEG_GRID_SIZE
        self.grid_w = grid_w or config.EEG_GRID_SIZE
        self.d_model = d_model or config.D_MODEL
        self.num_tokens = self.grid_h * self.grid_w

    def build(self, input_shape):
        # Linear projection from 1 (channel value) to d_model
        self.projection = layers.Dense(self.d_model, use_bias=True)
        # Precomputed 2D positional encoding (stored as non-trainable weight for Keras 3 compat)
        pe = create_2d_positional_encoding(self.grid_h, self.grid_w, self.d_model)
        self.pos_encoding = self.add_weight(
            name="pos_encoding", shape=pe.shape,
            initializer=tf.keras.initializers.Constant(pe),
            trainable=False)
        super().build(input_shape)

    def call(self, x):
        """
        Args:
            x: (batch, timesteps, grid_h, grid_w) tensor.
        Returns:
            (batch, timesteps, num_tokens, d_model) tensor.
        """
        batch_size = tf.shape(x)[0]
        timesteps = tf.shape(x)[1]

        # Flatten grid: (batch, timesteps, grid_h * grid_w)
        x = tf.reshape(x, [batch_size, timesteps, self.num_tokens])
        # Add feature dim: (batch, timesteps, num_tokens, 1)
        x = tf.expand_dims(x, -1)
        # Project to d_model: (batch, timesteps, num_tokens, d_model)
        x = self.projection(x)
        # Add positional encoding (broadcast over batch and timesteps)
        x = x + self.pos_encoding[tf.newaxis, tf.newaxis, :, :]
        return x

    def get_config(self):
        base = super().get_config()
        base.update({
            "grid_h": self.grid_h,
            "grid_w": self.grid_w,
            "d_model": self.d_model,
        })
        return base


class TemporalPositionEncoding(layers.Layer):
    """
    Standard 1D sinusoidal positional encoding for temporal sequences.

    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model) with positional encoding added.
    """

    def __init__(self, max_len=512, d_model=None, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model or config.D_MODEL

    def build(self, input_shape):
        pe = np.zeros((1, self.max_len, self.d_model), dtype=np.float32)
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)
        self.pos_encoding = self.add_weight(
            name="pos_encoding", shape=pe.shape,
            initializer=tf.keras.initializers.Constant(pe),
            trainable=False)
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        base = super().get_config()
        base.update({"max_len": self.max_len, "d_model": self.d_model})
        return base
