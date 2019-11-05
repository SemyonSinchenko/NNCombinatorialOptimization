"""
@author Semyon Sinchenko
"""

from collections import deque
from functools import partial
from random import randint
import tensorflow as tf

from .linalg_opp import moore_penrose_invert


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


def get_random_state_tensor(num_nodes):
    """Generate random state.
    
    Arguments:
        num_nodes {int} -- number of nodes
    
    Returns:
        tf.Tensor -- state
    """

    return tf.cast(tf.random.uniform((num_nodes,), 0, 2, tf.int32) * 2 - 1, dtype=tf.float32)


@tf.function(experimental_relax_shapes=True)
def get_state_probability(state, network):
    return network(tf.expand_dims(state, 0))


@tf.function(experimental_relax_shapes=True)
def get_acceptance_prob(state, new_state, network):
    return tf.stop_gradient(
        tf.square(get_state_probability(new_state, network)) / tf.square((get_state_probability(state, network)))
    )


def generate_samples(problem_dim, network, num_samples, drop_first):
    state = get_random_state_tensor(problem_dim)
    samples = deque()
    accepted = tf.constant(0.0)

    for _ in range(num_samples):
        n = tf.constant(randint(0, problem_dim))
        permuted = swap_node_in_state(state, n)
        accept_prob = get_acceptance_prob(state, permuted, network)
        
        if accept_prob >= tf.random.uniform((1, 1), 0.0, 1.0, tf.float32):
            accepted += tf.constant(1.0, tf.float32)
            state = permuted
            
        samples.append(state)
        

    for _ in range(drop_first):
        samples.popleft()

    return (tf.stack(samples), accepted)


@tf.function(experimental_relax_shapes=True)
def estimate_energy_of_state(state, extended_edge_list):
    return tf.reduce_sum(
        tf.math.abs(tf.sparse.reduce_sum(extended_edge_list * tf.expand_dims(state, 0), axis=1)) - 2.0
    ) / 2.0


@tf.function
def estimate_stochastic_reconfiguration_matrix(derivs, num_samples, l2):
    e_of_prod = tf.einsum("ki,kj", derivs, derivs) / num_samples
    avg_deriv = tf.reduce_mean(derivs, axis=0, keepdims=True)
    prod_of_e = tf.einsum("ki,kj", avg_deriv, avg_deriv)
    
    reg_part = tf.linalg.diag(tf.ones((e_of_prod.shape[0], ), tf.float32) * l2)
    
    return e_of_prod - prod_of_e + reg_part


@tf.function
def estimate_stochastic_gradients(derivs, energies, num_samples, l2):
    SS = estimate_stochastic_reconfiguration_matrix(derivs, num_samples, l2)
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
def update_weights_step(samples, network, edge_ext, optimizer, num_layers, n_samples, l2):
    network_outputs, grads = tf.vectorized_map(partial(get_out_and_grad, network=network), samples)
    network_outputs = tf.reshape(tf.stack(network_outputs), (n_samples, 1))
    energies = tf.map_fn(
        partial(estimate_energy_of_state, extended_edge_list=edge_ext),
        samples,
        tf.float32,
        parallel_iterations=n_samples
    )

    new_grads = []
    for i in range(num_layers):
        # i - layer
        w_shape = grads[i* 2].shape
        
        weights_and_biases = tf.concat(
            [tf.reshape(grads[i * 2], (n_samples, -1)), tf.reshape(grads[i * 2 + 1], (n_samples, -1))],
            axis=1
        )
        
        new_weights_and_biases = estimate_stochastic_gradients(
            weights_and_biases / network_outputs,
            energies, n_samples, l2
        )
        
        new_weights, new_biases = tf.split(new_weights_and_biases * 2.0, [w_shape[1] * w_shape[2], w_shape[2]], axis=0)
        new_grads.append(new_weights)
        new_grads.append(new_biases)
        
    optimizer.apply_gradients(
        ((tf.reshape(g, weights.shape)), weights) for g, weights in zip(new_grads, network.trainable_variables)
    )

    return energies


def learning_step(problem_dim, network, num_samples, drop_first, edge_ext, optimizer, num_layers, l2):
    samples, accepted = generate_samples(problem_dim, network, num_samples, drop_first)
    num_real_samples = num_samples - drop_first

    energies = update_weights_step(samples, network, edge_ext, optimizer, num_layers, num_real_samples, l2)

    return energies, accepted / tf.constant(num_samples, tf.float32)
