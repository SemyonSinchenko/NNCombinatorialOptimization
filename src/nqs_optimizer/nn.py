"""
@author Semyon Sinchenko
"""

from collections import deque
from functools import partial
from random import randint
import tensorflow as tf


@tf.function(experimental_relax_shapes=True)
def swap_node_in_state(state, n):
    """Given a state and N swap N-th node in state. Do not modify input state.
    
    Arguments:
        state {tf.Tensor} -- state
        n {int} -- node number
    
    Returns:
        tf.Tensor -- new state
    """
    return tf.multiply(state, tf.where(tf.range(0, state.shape[0], 1) == n, -1.0, 1.0))


@tf.function(experimental_relax_shapes=True)
def permute_tensor(state):
    """Generate permuted state by swap a random node.
    
    Arguments:
        state {tf.Tensor} -- state
    
    Returns:
        tf.Tensor -- new state
    """
    n = randint(0, state.shape[0])

    return swap_node_in_state(state, n)


def get_random_state_tensor(num_nodes):
    return tf.cast(tf.random.uniform((num_nodes,), 0, 2, tf.int32) * 2 - 1, dtype=tf.float32)


@tf.function(experimental_relax_shapes=True)
def get_state_probability(state, network):
    return network(tf.expand_dims(state, 0))


@tf.function(experimental_relax_shapes=True)
def get_acceptance_prob(state, new_state, network):
    return tf.stop_gradient(get_state_probability(new_state, network) / (get_state_probability(state, network) + 1e-32))


@tf.function(experimental_relax_shapes=True)
def get_new_state(state, new_state, network):
    accept_prob = get_acceptance_prob(state, new_state, network)
    if accept_prob >= tf.random.uniform((1, 1)):
        return new_state
    else:
        return state


def generate_samples(problem_dim, network, num_samples, drop_first):
    state = get_random_state_tensor(problem_dim)
    samples = deque()

    for _ in range(num_samples):
        permuted = permute_tensor(state)
        state = get_new_state(state, permuted, network)
        samples.append(state)

    for _ in range(drop_first):
        samples.popleft()

    return tf.stack(samples)


@tf.function(experimental_relax_shapes=True)
def estimate_energy_of_state(state, extended_edge_list):
    return tf.reduce_sum(tf.sparse.reduce_sum(extended_edge_list * tf.expand_dims(state, 0), axis=1) - 2.0) / 2.0


@tf.function
def estimate_stochastic_reconfiguration_matrix(derivs, l1):
    e_of_prod = tf.einsum("ij,jk -> ik", tf.linalg.adjoint(derivs), derivs)
    avg_deriv = tf.reduce_mean(derivs, axis=0, keepdims=True)
    prod_of_e = tf.matmul(tf.linalg.adjoint(avg_deriv), avg_deriv)
    
    SS = e_of_prod - prod_of_e
    reg_part = tf.eye(SS.shape[0], SS.shape[0]) * l1
    
    return SS + reg_part


@tf.function
def estimate_stochastic_gradients(derivs, energies, outputs, l1):
    SS = estimate_stochastic_reconfiguration_matrix(derivs, l1)
    e_of_prod = tf.reduce_mean(tf.multiply(tf.expand_dims(energies, 1), derivs), axis=0, keepdims=True)
    prod_of_e = tf.reduce_mean(derivs, axis=0, keepdims=True) * tf.reduce_mean(energies)
    
    forces = e_of_prod - prod_of_e
    stochastic_gradients = tf.linalg.cholesky_solve(SS, tf.linalg.adjoint(forces))

    return stochastic_gradients


@tf.function
def get_out_and_grad(state, network):
    o = get_state_probability(state, network)
    g = tf.gradients(o, network.trainable_variables)

    return o, g


@tf.function
def update_weights_step(samples, network, edge_ext, optimizer, num_layers, l1, n_samples):
    network_outputs, grads = tf.vectorized_map(partial(get_out_and_grad, network=network), samples)
    network_outputs = tf.expand_dims(tf.stack(network_outputs), 1)
    energies = tf.map_fn(
        partial(estimate_energy_of_state, extended_edge_list=edge_ext),
        samples,
        tf.float32,
        parallel_iterations=n_samples
    )

    new_grads = []
    for i in range(num_layers):
        # i - layer
        for j in range(2):
            # j==0: weights; j==1: biases
            new_grads.append(
                estimate_stochastic_gradients(
                    tf.reshape(grads[i* 2 + j], (n_samples, -1)),
                    energies,
                    network_outputs,
                    l1
                )
            )
        
    optimizer.apply_gradients(
        ((tf.reshape(g, weights.shape)), weights) for g, weights in zip(new_grads, network.trainable_variables)
    )

    return energies


def learning_step(problem_dim, network, num_samples, drop_first, edge_ext, optimizer, num_layers, l1):
    samples = generate_samples(problem_dim, network, num_samples, drop_first)
    num_real_samples = num_samples - drop_first
    return update_weights_step(
        samples, network, edge_ext, optimizer, num_layers, l1, num_real_samples
    )
