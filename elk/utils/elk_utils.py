"""
elk_utils.py
helper functions for ELK
"""

import jax.numpy as jnp

# packing and unpacking code (useful for getting Jacobian)
def pack_state(state):
    """
    Args:
        state: tuple, contains the state of the model
               right now, it has two elements
               the first it the output
               the second is a list of the hidden state (prev)
    """
    x, prev_states = state
    return jnp.hstack((x, jnp.concatenate(prev_states)))


def make_unpacking_tuple(model):
    """
    Creates the appropriate unpacking tuple for the model (of type chained AF)

    Args:
        * model: should be of type chained_AF

    Returns:
        * tuple, to divide a packed state into first the outputs, then the hidden states
    """
    P = model.conditioners[0].seq_model.ssm.P
    n_layers = model.conditioners[0].seq_model.n_layers

    unpacking_lst = [1] + [P for _ in range(n_layers - 1)]
    unpacking_tple = tuple(jnp.cumsum(jnp.array(unpacking_lst)).tolist())

    return unpacking_tple


def unpack_state(unpacking_tuple, packed_state):
    """
    unpack a packed states according to an unpacking tuple

    Args:
        * unpacking tuple (output first, then hiddnes)
        * packed_state
    """
    unpacked_full = jnp.split(packed_state, unpacking_tuple)
    prev_states = unpacked_full[1:]
    x = unpacked_full[0]
    return x.real, prev_states


def packed_single_step(model, unpacking_tuple, packed_state, noise):
    """
    Like `chained_AF.single_step`, but takes in a packed state
    A packed state has everything in the Markovian state as a single vector

    Args:
        * model: should be of type chained_AF
        * packed_state: jax.Array, the packed state
        * noise: jax.Array, noise from the base dist

    Returns:
        * jax.Array, the packed new state
    Notes:
        * accepting a scalar input
    """
    input, prev_states = unpack_state(unpacking_tuple, packed_state)

    new_states, out = model.single_step(prev_states, input, noise)
    return pack_state((out, new_states))
