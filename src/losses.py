"""
Focal Loss for multi-class classification.

Focal loss down-weights well-classified examples, focusing training
on hard, misclassified samples — particularly effective for the
imbalanced 3-class (Low/Medium/High) emotion labels where the
Medium class (single score value = 3) is underrepresented.

Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
"""

import tensorflow as tf

from src import config


class CategoricalFocalLoss(tf.keras.losses.Loss):
    """
    Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples.
               gamma=0 is equivalent to standard cross-entropy. (default: 2.0)
        alpha: Per-class weight array of shape (num_classes,), or None for
               uniform weighting. (default: None)
        label_smoothing: Smoothing factor for one-hot labels. (default: 0.0)
    """

    def __init__(self, gamma=None, alpha=None, label_smoothing=None,
                 name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma if gamma is not None else config.FOCAL_LOSS_GAMMA
        self.alpha = alpha  # Set later if None, via set_alpha_from_weights
        self.label_smoothing = (label_smoothing if label_smoothing is not None
                                else config.LABEL_SMOOTHING)

    def set_alpha_from_weights(self, class_weights):
        """
        Set per-class alpha weights from the class_weights dict.

        Normalizes weights so they sum to num_classes (preserving relative
        magnitude while keeping loss scale consistent).

        Args:
            class_weights: dict {class_idx: weight}.
        """
        num_classes = len(class_weights)
        alpha = [class_weights.get(i, 1.0) for i in range(num_classes)]
        total = sum(alpha)
        self.alpha = [a * num_classes / total for a in alpha]

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: One-hot encoded labels (batch, num_classes).
            y_pred: Predicted probabilities (batch, num_classes).
        Returns:
            Scalar loss value.
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + (
                self.label_smoothing / num_classes
            )

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Cross-entropy component: -y_true * log(y_pred)
        ce = -y_true * tf.math.log(y_pred)

        # Focal modulation: (1 - p_t)^gamma
        modulating_factor = tf.pow(1.0 - y_pred, self.gamma)

        # Apply focal modulation
        focal_loss = modulating_factor * ce

        # Apply per-class alpha weights
        if self.alpha is not None:
            alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
            focal_loss = focal_loss * alpha_tensor[tf.newaxis, :]

        # Sum over classes, mean over batch
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        base = super().get_config()
        base.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing,
        })
        return base
