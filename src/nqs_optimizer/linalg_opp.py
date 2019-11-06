"""
@author Semyon Sinchenko
"""

from functools import partial
import tensorflow as tf


@tf.function
def __divide_on_by(x, threshold):
    if x >= threshold:
        return tf.constant(1.0) / x
    else:
        return tf.constant(0.0)


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
    
    threshold = tf.math.reduce_max(s) * threshold
    s_inv = tf.linalg.diag(tf.vectorized_map(partial(__divide_on_by, threshold=threshold), s))

    return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))
