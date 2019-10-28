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
    p_other = get_state_probability(permuted, network)

    return p_other / (p_state + 1e-32)


@tf.function(experimental_relax_shapes=True)
def estimate_superposition_part(adjacency, permutation_probs, state):
    return tf.reduce_sum(tf.multiply(tf.multiply(adjacency, (-state)), tf.reshape(permutation_probs, (1, -1))))

@tf.function(experimental_relax_shapes=True)
def estimate_local_energy_of_state(state, network, edge_list, adjacency, num_nodes):
    energy = estimate_energy_of_state(state, edge_list)
    p_state = get_state_probability(state, network)
    all_permutaions = tf.map_fn(
        partial(get_permutation_value, p_state=p_state, state=state, network=network),
        tf.range(0, num_nodes, dtype=tf.int32)
    )

    superposition = estimate_superposition_part(adjacency, all_permutaions, state)

    return (energy + superposition) / 2.0


def estimate_local_energies(samples, network, edge_list, adjacency):
    local_energy_of_state_closure = partial(
        estimate_local_energy_of_state, network=network, edge_list=edge_list, adjacency=adjacency
    )
    return tf.map_fn(local_energy_of_state_closure, samples, dtype=tf.float32, parallel_iterations=50)


@tf.function
def update_weights_step(samples, network, edge_list, adjacency, optimizer, loss):
    with tf.GradientTape() as tape:
        energies = estimate_local_energies(samples, network, edge_list, adjacency)
        grads = tape.gradient(energies, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))

    return energies


def learning_step(problem_dim, network, num_samples, drop_first, edge_list, adjacency, optimizer, loss):
    samples = generate_samples(problem_dim, network, num_samples, drop_first)
    return update_weights_step(samples, network, edge_list, adjacency, optimizer, loss)
