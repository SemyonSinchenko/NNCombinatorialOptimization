"""
@author Semyon Sinchenko
"""

import tensorflow as tf
from .nn import *

@tf.function(experimental_relax_shapes=True)
def estimate_energy_of_state(state, edge_list):
    return tf.reduce_sum(tf.map_fn(lambda edge: state[edge[0]] * state[edge[1]] - 1.0, edge_list, dtype=tf.float32))


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
            adjacency * tf.multiply(permutation_probs, -state)
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