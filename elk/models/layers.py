"""
layers.py
contains a single SequenceLayer, which applies the ssm, then a non-linearity
adapted from: https://github.com/lindermanlab/S5
"""

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

from typing import Optional


class SequenceLayer(eqx.Module):
    """Defines a single S5 layer, with S5 SSM, nonlinearity,
        dropout, batch/layer norm, etc.

    Can also take in an RNN
    Args:
        key         (jr.PRNGKey)
        ssm         (eqx.Module): the SSM to be used (i.e. S5 ssm). Can also take in an RNN
        dropout     (float32):  dropout rate
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                                we usually refer to this size as H
        activation  (string):   Type of activation function to use
        training    (bool):     whether in training mode or not
        prenorm     (bool):     apply prenorm if true or postnorm if false

        Notes: for now, just use LayerNorm
    """

    ssm: eqx.Module  # can be an ssm or an rnn
    dropout: float
    d_model: float
    activation: str = eqx.field(static=True)
    training: bool
    prenorm: bool
    skip: bool
    postnorm: bool
    out1: Optional[eqx.nn.Linear]
    out2: Optional[eqx.nn.Linear]
    norm: eqx.nn.LayerNorm
    drop: eqx.nn.Dropout
    ssm_type: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        ssm,
        dropout,
        d_model,
        activation,
        training=True,
        prenorm=False,
        skip=True,
        postnorm=True,
        step_rescale=1.0,
    ):
        """Initializes the ssm, batch/layer norm and dropout"""
        out1_key, out2_key = jr.split(key, 2)
        self.ssm = ssm
        self.ssm_type = self.ssm.ssm_type
        self.dropout = dropout
        self.d_model = d_model
        self.activation = activation
        self.training = training
        self.prenorm = prenorm
        self.skip = skip
        self.postnorm = postnorm

        if self.activation in ["full_glu"]:
            self.out1 = eqx.nn.Linear(self.d_model, self.d_model, key=out1_key)
            self.out2 = eqx.nn.Linear(self.d_model, self.d_model, key=out2_key)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out1 = None
            self.out2 = eqx.nn.Linear(self.d_model, self.d_model, key=out2_key)
        else:
            self.out1 = None
            self.out2 = None

        self.norm = eqx.nn.LayerNorm((d_model,))  # we do need to vmap over the seq len

        # note that we want the same dropout over the sequence length L
        # which is the first axes in the data layout
        self.drop = eqx.nn.Dropout(self.dropout, inference=not self.training)

    def nonlinearity(self, x, key):
        """
        x: the output, jax.Array, dimension (H,)
        """
        if self.activation in ["full_glu"]:
            this_key, key = jr.split(key)
            x = self.drop(jnn.gelu(x), key=this_key)
            x = self.out1(x) * jnn.sigmoid(self.out2(x))
            this_key, key = jr.split(key)
            x = self.drop(x, key=this_key)
        elif self.activation in ["half_glu1"]:
            this_key, key = jr.split(key)
            x = self.drop(jnn.gelu(x), key=this_key)
            x = x * jnn.sigmoid(self.out2(x))
            this_key, key = jr.split(key)
            x = self.drop(x, key=this_key)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            this_key, key = jr.split(key)
            x1 = self.drop(jnn.gelu(x), key=this_key)
            x = x * jnn.sigmoid(self.out2(x1))
            this_key, key = jr.split(key)
            x = self.drop(x, key=this_key)
        elif self.activation in ["gelu"]:
            this_key, key = jr.split(key)
            x = self.drop(jnn.gelu(x), key=this_key)
        elif self.activation in ["identity"]:
            x = self.drop(x, key=key)
        else:
            raise NotImplementedError(
                "Activation: {} not implemented".format(self.activation)
            )
        return x

    def __call__(self, x, integration_timesteps, drop_key):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, H)
             integration_timesteps
             drop_key (jr.PRNGKey): randomness for dropout
        Returns:
            output sequence (float32): (L+1, H)

        Note that S5 now increases the length by 1 because of the phantom first input
        """
        if self.ssm.use_x0:
            skip = jnp.vstack([self.ssm.x0, x])
        else:
            skip = x

        if self.prenorm:
            x = jax.vmap(self.norm)(x)  # vmap over the seq len

        # apply the SSM
        if self.ssm_type in ["ssm"]:
            x = self.ssm(x, integration_timesteps)  # (L+1,)
        elif self.ssm_type in ["rnn"]:
            x = self.ssm(x)
        else:
            raise NotImplementedError(f"SSM type not implemented")

        # apply a non-linearity
        drop_keys = jr.split(drop_key, x.shape[0])
        x = jax.vmap(self.nonlinearity)(
            x, drop_keys
        )  # we need to vmap over the sequence length

        # skip connection
        if self.skip:
            x = skip + x

        # LayerNorm
        if self.postnorm:
            x = jax.vmap(self.norm)(x)  # vmap over the seq len

        return x

    def single_step(self, state, input, drop_key):
        """
        applies to a single step, so no need to vmap over the sequence length

        state: jax.Array, dimension (P,)
        input: jax.Array, dimension (H,)
        """
        skip = input

        if self.prenorm:
            input = self.norm(input)

        hidden_state, x = self.ssm.single_step(
            state, input
        )  # 

        x = self.nonlinearity(x, drop_key)

        if self.skip:
            x = x + skip
        if self.postnorm:
            x = self.norm(x)

        return hidden_state, x

    def show_hiddens(self, input_sequence, drop_key):
        """
        Inputs
        ------
        input_sequence (L,H)

        Outputs
        -------
        hs: hiddens (L,P)
        ys: outputs (L,H) [after non-linearity]
        """
        skip = input_sequence
        if self.prenorm:
            x = jax.vmap(self.norm)(input_sequence)
        else:
            x = input_sequence
        hs, ys = self.ssm.show_hiddens(x)
        # apply a non-linearity
        drop_keys = jr.split(drop_key, ys.shape[0])
        ys = jax.vmap(self.nonlinearity)(
            ys, drop_keys
        )  # we need to vmap over the sequence length
        if self.skip:
            if self.ssm.use_x0:
                skip = jnp.vstack([self.ssm.x0, input_sequence])
            ys = skip + ys
        if self.postnorm:
            ys = jax.vmap(self.norm)(ys)  # vmap over the seq len
        return hs, ys
