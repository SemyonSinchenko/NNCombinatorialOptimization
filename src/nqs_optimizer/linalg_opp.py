"""
@author Semyon Sinchenko
"""

from functools import partial
import tensorflow as tf


@tf.function
def moore_penrose_invert(tensor, threshold=tf.constant(1.0e-5)):
    """Moore-Penrose pseudo-ivertion of given tensor.
    
    Arguments:
        tensor {tf.Tensor} -- matrix
    
    Keyword Arguments:
        threshold {tf.float32} -- drop singular values less than max_s * threshold (default: {1.0e-4})
    
    Returns:
        tf.Tensor -- inverted matrix
    """

    s, u, v = tf.linalg.svd(tensor)
    
    threshold_ = tf.math.reduce_max(s) * threshold
    s_no_zeros = tf.boolean_mask(s, s > threshold_)
    s_inv = tf.linalg.diag(
        tf.concat([tf.math.reciprocal(s_no_zeros), tf.zeros(tf.size(s) - tf.size(s_no_zeros), tf.float32)], axis=0)
    )

    return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))
