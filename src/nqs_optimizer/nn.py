"""
@author Semyon Sinchenko

JIT-compiled functions.
"""

from collections import deque
from functools import partial

import tensorflow as tf


@tf.function
def swap_node_in_state(state, n):
    """Given a state and N swap N-th node in state. Do not modify input state.
    
    Parameters
    ----------
    state : tf.Tensor
        state
    n : int
        node number
    
    Returns
    -------
    tf.Tensor
        new state
    """
    return tf.multiply(state, tf.where(tf.range(0, state.shape[0], 1) == n, -1.0, 1.0))


@tf.function
def estimate_energy_of_state(state, extended_edge_list):
    """Estimate energy of given state. JIT-compiled function.
    Here we use the following formula:
        E = Sum_{rows} (|Sum_{i} (edge_list_i * state_i)| - 2) / 2

    That formula is equal to by-row computation but it can be effectively realized if Tensorflow.
    It is significant more effective than Carleo-approach based on Hxx-matrix because here we use
    the fact of diagonalization of Hxx in Z-basis.
    
    Parameters
    ----------
    state : tf.Tensor
        State.
    extended_edge_list : tf.Tensor
        Sparse tensor with number of rows equals to the number of edges in Graph.
        Each row of that tensor contain only two non-zero values. That values are
        in positions of start node and end node of corresponded to that row edge.
    
    Returns
    -------
    tf.float32
        Energy of state.
    """
    return tf.reduce_sum(
        tf.math.abs(tf.sparse.reduce_sum(extended_edge_list * tf.expand_dims(state, 0), axis=1)) - 2.0
    ) / 2.0


@tf.function
def estimate_stochastic_reconfiguration_matrix(derivs, num_samples, l2):
    """Compute regulirized stochastic reconfiguration matrix S of partial derivatives.
    
    Parameters
    ----------
    derivs : tf.Tensor
        Tensor of derivatives that has shape (n_samples x n_wights)
    num_samples : tf.int32
        Number of real samples.
    l2 : tf.float32
        Regularization L2-lambda.
    
    Returns
    -------
    tf.Tensor
        Matrix S.
    """
    e_of_prod = tf.einsum("ki,kj", derivs, derivs) / num_samples
    avg_deriv = tf.reduce_mean(derivs, axis=0, keepdims=True)
    prod_of_e = tf.einsum("ki,kj", avg_deriv, avg_deriv)
    
    SS = e_of_prod - prod_of_e

    # Compute regularization part
    reg_part = tf.linalg.diag(tf.ones((SS.shape[0], ), tf.float32) * l2)
    
    return SS + reg_part


@tf.function
def estimate_stochastic_gradients(derivs, energies, num_samples, l2):
    """Compute stochastic derivatives (aka Natural gradient)
    
    Parameters
    ----------
    derivs : tf.Tensor
        Tensor of derivatives that has shape (n_samples x n_wights)
    energies : tf.Tensor
        Tensor with energies that has shape (n_samples x 1)
    num_samples : tf.int32
        Number of real samples.
    l2 : tf.float32
        Regularization L2-lambda.
    
    Returns
    -------
    tf.Tensor
        Natural derivatives.
    """
    SS = estimate_stochastic_reconfiguration_matrix(derivs, num_samples, l2)

    # Compute forces
    e_of_prod = tf.reduce_mean(tf.multiply(tf.expand_dims(energies, 1), derivs), axis=0, keepdims=True)
    prod_of_e = tf.reduce_mean(derivs, axis=0, keepdims=True) * tf.reduce_mean(energies)
    forces = e_of_prod - prod_of_e

    # Compute (S^-1 * F)
    return tf.linalg.cholesky_solve(SS, tf.linalg.adjoint(forces))


@tf.function
def get_out_and_grad(state, network):
    """Get output of network and gradient of output by weights.
    
    Parameters
    ----------
    state : tf.Tensor
        State.
    network : tf.keras.Sequential
        NN.
    
    Returns
    -------
    (tf.float32, List[tf.Tenor])
        Output of NN and gradients.
    """
    o = network(tf.expand_dims(state, 0))
    g = tf.gradients(o, network.trainable_variables)

    return tf.squeeze(o), g
