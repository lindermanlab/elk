# model_utils.py

import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

from jax.nn.initializers import normal

import json

import argparse


# Linear layer with a sensible initialization.
class SmallWeightLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key, scale=0.1):
        wkey, bkey = jax.random.split(key)
        self.weight = (
            jr.normal(wkey, (out_size, in_size))
            * scale
            * jnp.sqrt(2.0 / (in_size + out_size))
        )
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias


# Diagonal feedthrough matrix that can be turned on or off
class OptionalMultiply(eqx.Module):
    diag_mat: jax.Array
    flag: bool

    def __init__(self, size, flag, key):
        """
        size: integer length of the diagonal matrix
        flag: bool
                True: the matrix multiplication acutally happens
                False: null op (return 0)
        key: jr.PRNGKey
        """
        self.flag = flag
        self.diag_mat = normal(stddev=1.0)(key, (size,))

    def __call__(self, u):
        if self.flag:
            return self.diag_mat * u
        else:
            return jnp.zeros_like(u)


# rectifying non-linearity
def rectify(log_scale, softplus=True):
    if softplus:
        scale = jnn.softplus(log_scale)
    else:
        scale = jnp.exp(log_scale)
    return scale


# helper code for saving equinox modules
def save_eqx_module(filename, hyperparams, model):
    """
    Built with the paradigm of using a make function
    So need to provide the hyperparams that will be given to the make function
    """
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


# helper code for parsing
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def sample_AR_many_layers(model, length, key=jr.PRNGKey(0), dtype=jnp.complex64):
    """
    Handles skip connections and feedthroughs

    all hidden states same size

    only dealing with one conditioner, using a StackedEncoder model that can have many layers
    """
    f5 = model.conditioners[0]
    softplus = model.softplus
    us = jr.normal(key, (length,))
    P = f5.seq_model.ssm.P
    n_layers = f5.seq_model.n_layers
    h_minus_ones = [
        jnp.zeros(P, dtype=dtype) for _ in range(n_layers)
    ]  # dummy to get single step rolling

    x0 = f5.seq_model.layers[0].ssm.x0  # (H,)
    # x0 edge case
    h0, out_0 = f5.seq_model.single_step(h_minus_ones, x0, no_encode=True)
    mu, log_scale = f5.decoder(out_0)
    scale = rectify(log_scale, softplus=softplus)
    x1 = mu + scale * us[0]
    x1 = jnp.reshape(x1, (1,))

    def f(state, u):
        x, hs = state
        new_hiddens, output = f5.seq_model.single_step(hs, x)
        mu, log_scale = f5.decoder(output)
        scale = rectify(log_scale, softplus=softplus)
        new_x = mu + scale * u
        new_x = jnp.reshape(new_x, (1,))
        return (new_x, new_hiddens), new_x

    start_state = (x1, h0)
    _, xs = jax.lax.scan(f, start_state, us[1:])

    return jnp.vstack([jnp.reshape(x1, (1, 1)), xs]), h0, us, start_state
