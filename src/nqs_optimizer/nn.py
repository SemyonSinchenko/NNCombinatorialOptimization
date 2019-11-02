"""
@author Semyon Sinchenko
"""

from collections import deque
from functools import partial
from random import randint
import tensorflow as tf


@tf.function(experimental_relax_shapes=True)
def swap_node_in_state(state, n):
    return tf.multiply(state, tf.where(tf.range(0, state.shape[0], 1) == n, -1.0, 1.0))


@tf.function(experimental_relax_shapes=True)
def permute_tensor(state):
    n = randint(0, state.shape[0])

    return swap_node_in_state(state, n)


def get_random_state_tensor(num_nodes):
    return tf.cast(tf.random.uniform((num_nodes,), 0, 2, tf.int32) * 2 - 1, dtype=tf.float32)


@tf.function(experimental_relax_shapes=True)
def get_state_probability(state, network):
    return network(tf.reshape(state, (1, -1)))


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
def estimate_energy_of_state(state, edge_list):
    return tf.reduce_sum(tf.map_fn(lambda edge: state[edge[0]] * state[edge[1]] - 1, edge_list, dtype=tf.float32))


@tf.function(experimental_relax_shapes=True)
def get_permutation_value(pos, p_state, state, network):
    permuted = swap_node_in_state(state, pos)
    return tf.minimum(
        tf.stop_gradient(get_state_probability(permuted, network) / (p_state + 1e-32)),
        tf.constant(1.0)
    )

@tf.function(experimental_relax_shapes=True)
def estimate_superposition_part(adjacency, permutation_probs, state):
    return tf.stop_gradient(
        tf.sparse.reduce_sum(
            adjacency * tf.multiply(-state, tf.reshape(permutation_probs, (1, -1)))
        )
    )

@tf.function(experimental_relax_shapes=True)
def estimate_local_energy_of_state(state, network, edge_list, adjacency, num_nodes):
    energy = estimate_energy_of_state(state, edge_list)
    p_state = tf.stop_gradient(get_state_probability(state, network))
    all_permutaions = tf.vectorized_map(
        partial(get_permutation_value, p_state=p_state, state=state, network=network),
        tf.range(0, num_nodes, dtype=tf.int32)
    )

    superposition = estimate_superposition_part(adjacency, all_permutaions, state)

    return (energy + superposition) / 2.0

@tf.function
def estimate_local_energies(samples, network, edge_list, adjacency, num_nodes):
    local_energy_of_state_closure = partial(
        estimate_local_energy_of_state, network=network, edge_list=edge_list, adjacency=adjacency, num_nodes=num_nodes
    )
    return tf.map_fn(local_energy_of_state_closure, samples, dtype=tf.float32, parallel_iterations=500)

@tf.function
def estimate_all_real_energies(samples, edge_list):
    return tf.map_fn(
        partial(estimate_energy_of_state, edge_list=edge_list),
        samples,
        tf.float32,
        parallel_iterations=100
    ) / 2.0

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
    e_of_prod = tf.reduce_mean(tf.multiply(energies, derivs), axis=0, keepdims=True)
    prod_of_e = tf.reduce_mean(derivs, axis=0, keepdims=True) * tf.reduce_mean(energies)
    
    forces = e_of_prod - prod_of_e
    stochastic_gradients = tf.linalg.cholesky_solve(SS, tf.linalg.adjoint(forces))

    return stochastic_gradients

@tf.function
def get_network_gradients(samples, network, num_samples):
    network_outputs = tf.vectorized_map(
        partial(get_state_probability, network=network),
        samples
    )

    grads = [
        tf.gradients(net_output, network.trainable_variables) 
        for net_output in tf.unstack(network_outputs, num=num_samples)
    ]

    return (network_outputs, grads)
    

def update_weights_step(samples, network, edge_list, adjacency, optimizer, num_nodes, num_layers, l1, n_samples):
    energies = estimate_local_energies(samples, network, edge_list, adjacency, num_nodes)
    network_outputs, grads = get_network_gradients(samples, network, n_samples)

    new_grads = []
    for i in range(num_layers):
        for j in range(2):
            new_grads.append(
                estimate_stochastic_gradients(
                    tf.stack([tf.reshape(g_i[i * 2 + j], (-1, )) for g_i in grads]),
                    energies,
                    network_outputs,
                    l1
                )
            )
        
    optimizer.apply_gradients(zip(new_grads, network.trainable_variables))

    real_energies = estimate_all_real_energies(samples, edge_list)

    return (energies, real_energies)


def learning_step(problem_dim, network, num_samples, drop_first, edge_list, adjacency, optimizer, num_nodes, num_layers, l1):
    samples = generate_samples(problem_dim, network, num_samples, drop_first)
    num_real_samples = num_samples - drop_first
    return update_weights_step(
        samples, network, edge_list, adjacency, optimizer, num_nodes, num_layers, l1, num_real_samples
    )
