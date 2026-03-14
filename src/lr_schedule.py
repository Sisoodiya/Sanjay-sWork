"""
Learning rate schedule: linear warmup followed by cosine decay.

Transformer architectures benefit from a warmup phase to stabilize
early gradient updates, followed by smooth cosine annealing.
"""

import numpy as np
import tensorflow as tf

from src import config


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup for `warmup_steps`, then cosine decay to near zero.

    lr(step) =
        if step < warmup_steps:
            base_lr * (step / warmup_steps)          [linear warmup]
        else:
            base_lr * 0.5 * (1 + cos(π * progress))  [cosine decay]

    where progress = (step - warmup_steps) / (total_steps - warmup_steps).
    """

    def __init__(self, base_lr=None, warmup_steps=0, total_steps=1000,
                 min_lr=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.base_lr = base_lr or config.LEARNING_RATE
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.base_lr * (step / tf.maximum(warmup, 1.0))

        # Cosine decay
        progress = (step - warmup) / tf.maximum(total - warmup, 1.0)
        progress = tf.minimum(progress, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.cos(np.pi * progress)
        )

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


def build_lr_schedule(steps_per_epoch, epochs=None, warmup_epochs=None):
    """
    Create a WarmupCosineDecay schedule from epoch-based config.

    Args:
        steps_per_epoch: Number of training steps per epoch.
        epochs: Total training epochs (default: config.EPOCHS).
        warmup_epochs: Warmup epochs (default: config.LR_WARMUP_EPOCHS).
    Returns:
        WarmupCosineDecay schedule instance.
    """
    if epochs is None:
        epochs = config.EPOCHS
    if warmup_epochs is None:
        warmup_epochs = config.LR_WARMUP_EPOCHS

    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    return WarmupCosineDecay(
        base_lr=config.LEARNING_RATE,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
