"""
METRICS

Useful metrics during model training.

Author: Peter Lais
Last updated: 10/15/2022
"""

import tensorflow as tf
import voxelmorph as vxm

def ncc_metric(y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None) -> tf.Tensor:
    """
    Returns the NCC between the GRAYSCALE images y_true and y_pred.
    
    Follows the typical parameters of tf.keras.metrics.Metric.
    """
    if sample_weight is not None: raise NotImplementedError('ncc_metric does not support sample weights.')
    return vxm.losses.NCC().ncc(tf.math.reduce_mean(y_true, axis=-1, keepdims=True),
                     tf.math.reduce_mean(y_pred, axis=-1, keepdims=True))
ncc_metric.__name__ = 'ncc_metric'

def mi_metric(y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None) -> tf.Tensor:
    """
    Returns the MI between the GRAYSCALE images y_true and y_pred. Note
    that this metric is not explicitly normalized, so only relative trends
    in its value may be analyzed.

    Follows the typical parameters of tf.keras.metrics.Metric.
    """
    if sample_weight is not None: raise NotImplementedError('mi_metric does not support sample weights.')
    return -vxm.losses.MutualInformation().loss(tf.math.reduce_mean(y_true, axis=-1, keepdims=True),
                      tf.math.reduce_mean(y_pred, axis=-1, keepdims=True))
mi_metric.__name__ = 'mi_metric'

def dice_metric(y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None) -> tf.Tensor:
    """
    Returns the Dice score between the LABEL images y_true and y_pred.

    Follows the typical parameters of tf.keras.metric.Metric.
    """
    if sample_weight is not None: raise NotImplementedError('dice_metric does not support sample weights.')
    return -vxm.losses.Dice().loss(y_true, y_pred)
dice_metric.__name__ = 'dice_metric'