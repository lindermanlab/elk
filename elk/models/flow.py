"""
flow.py
functions for normalizing flows
"""


from tensorflow_probability.substrates import jax as tfp


tfb = tfp.bijectors
tfd = tfp.distributions

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

import equinox as eqx

from typing import List

import json

# imports for AffineFlowSoftPlus
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
from tensorflow_probability.substrates.jax.bijectors.masked_autoregressive import (
    _validate_bijector_fn,
)

from elk.models.encoders import init_StackedEncoder
from elk.models.model_utils import SmallWeightLinear, rectify


class AffineFlowSoftPlus(tfb.MaskedAutoregressiveFlow):
    """
    Modify the jax substrate of the tfb implementation of MaskedAutoregressiveFlow
    (you can find in the pip installation of tfp)

    Changelog
    ---------
    * replace log_scale with scale = jnn.softplus
    """

    def __init__(
        self,
        shift_and_log_scale_fn=None,
        bijector_fn=None,
        is_constant_jacobian=False,
        validate_args=False,
        unroll_loop=False,
        event_ndims=1,
        name=None,
    ):
        parameters = dict(locals())
        name = name or "masked_autoregressive_flow"
        with tf.name_scope(name) as name:
            self._unroll_loop = unroll_loop
            self._event_ndims = event_ndims
            if bool(shift_and_log_scale_fn) == bool(bijector_fn):
                raise ValueError(
                    "Exactly one of `shift_and_log_scale_fn` and "
                    "`bijector_fn` should be specified."
                )
            if shift_and_log_scale_fn:

                def _bijector_fn(x, **condition_kwargs):
                    params = shift_and_log_scale_fn(x, **condition_kwargs)
                    if tf.is_tensor(params):
                        shift, log_scale = tf.unstack(params, num=2, axis=-1)
                    else:
                        shift, log_scale = params

                    bijectors = []
                    if shift is not None:
                        bijectors.append(tfb.Shift(shift))
                    if log_scale is not None:
                        bijectors.append(
                            tfb.Scale(scale=jnn.softplus(log_scale))
                        )  # this is the change from source
                    return tfb.Chain(bijectors, validate_event_size=False)

                bijector_fn = _bijector_fn

            if validate_args:
                bijector_fn = _validate_bijector_fn(bijector_fn)
            # Still do this assignment for variable tracking.
            self._shift_and_log_scale_fn = shift_and_log_scale_fn
            self._bijector_fn = bijector_fn
            super(tfb.MaskedAutoregressiveFlow, self).__init__(
                forward_min_event_ndims=self._event_ndims,
                is_constant_jacobian=is_constant_jacobian,
                validate_args=validate_args,
                parameters=parameters,
                name=name,
            )


# Define  a simple class that has an S5 encoder and a linear decoder.
class EncDec(eqx.Module):
    """
    This is the model class we use for our conditioners (i.e., the functions that produces shifts and scales)

    The point of this object is to allow the sequence model to have arbitrary hidden dimensons
    But then to compress the output down to 2 dimenions (mean and log scale) for use as a conditioner in an AF

    Note: now that you have modified S5 to have a phantom first state u0, you don't need to do the off-by-one shifting
    """

    seq_model: eqx.Module  # StackedEncoderModel
    decoder: eqx.Module
    # prior_params: jax.Array

    def __init__(self, seq_model, decoder):
        self.seq_model = seq_model
        self.decoder = decoder

    def __call__(self, x, integration_timesteps=None, drop_key=jr.PRNGKey(0)):
        y = self.seq_model(x, integration_timesteps, drop_key)  # (L+1,)
        unshifted_seq = jax.vmap(self.decoder)(y)
        output = unshifted_seq[:-1]
        return output


class chained_AF(eqx.Module):
    """
    AF made from chaining many affine transformations together

    Inputs
    ------
    input_length: int, length of input sequence L
    conditioners: List[eqx.Module], list of the conditioners across the layers

    Notes
    -----
    * gives the same results as you sample AR modules as long as you account for
        * tfd.MVN and jr.normal being the same (currently, they are)
        * splitting of the random seed
    """

    input_length: int  # length of input sequence L
    conditioners: List[eqx.Module]  # EncDec
    softplus: bool

    def __init__(self, input_length, conditioners, softplus=False):
        self.input_length = input_length
        self.conditioners = conditioners
        self.softplus = softplus

    def get_transformation(self):
        transformations = []
        for c in self.conditioners[::-1]:
            if self.softplus:
                af = AffineFlowSoftPlus(shift_and_log_scale_fn=c)
            else:
                af = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=c)
            transformations.append(af)
        return tfb.Chain(transformations)

    def get_dist(self):
        base_dist = tfd.MultivariateNormalDiag(
            loc=jnp.zeros(self.input_length), scale_diag=jnp.ones(self.input_length)
        )
        return tfd.TransformedDistribution(
            distribution=base_dist, bijector=self.get_transformation()
        )

    def log_prob(self, x):
        dist = self.get_dist()
        return dist.log_prob(x)

    def sample(self, batch_size, key):
        keys = jr.split(key, batch_size)  # notice that you are splitting in this line

        def single_sample(key):
            dist = self.get_dist()
            return dist.sample(seed=key)

        return jax.vmap(single_sample)(keys)

    def single_step(self, prev_states, input, noise):
        """
        Single step of the F5

        Args:
            prev_states: list[jax.Array], all the hidden ssm states
            input: jax.Array, the current input (data space)
            noise: jax.Array, noise from the base dist

        Returns:
            new_states: list[jax.Array], the new hidden states
            new_x: jax.Array, the new data point

        Notes:
            *only works if there is only one conditioner
            *always encoding the input
        """
        new_states, out = self.conditioners[0].seq_model.single_step(prev_states, input)
        # now we need to decode out
        mu, log_scale = self.conditioners[0].decoder(out)
        scale = rectify(log_scale, softplus=self.softplus)
        new_x = mu + scale * noise
        return new_states, new_x


def make_f5(
    *,
    key,
    data_dim,
    model_Ps,
    stack_length,
    n_layers,
    dec_scale,
    skip,
    use_D,
    softplus,
    ssm_type="ssm",
):
    """
    data_dim: sequence length
    models_Ps: list of ints, length is stack_length, gives the SSM dimension over stacked f5s
    stack_length: how many f5s to stack
    n_layers: how many layers to use in each f5
    ssm_type: str, either 'ssm' or 'rnn'
        if 'rnn', build out a 1 layer stacked encoder with a GRU sequence model

    Notes:
        - the * at the start of the args indicates that only kwargs are ok
    """
    model_keys = jr.split(key, stack_length)
    conditioners = []
    encs, decs = [], []
    for k, P in zip(model_keys, model_Ps):
        k1, k2 = jr.split(k)
        if ssm_type == "ssm":
            enc = init_StackedEncoder(
                k1,
                n_layers=n_layers,
                num_features=1,
                P=P,
                H=P,
                skip=skip,
                use_D=use_D,
                ssm_type=ssm_type,
            )
        elif ssm_type == "rnn":
            enc = init_StackedEncoder(
                k1, n_layers=1, num_features=1, P=1, H=P, skip=skip, ssm_type=ssm_type
            )
        else:
            raise NotImplementedError(f"SSM type {ssm_type} not implemented")
        dec = SmallWeightLinear(P, 2, key=k2, scale=dec_scale)
        conditioners.append(EncDec(enc, dec))
        encs.append(enc)
        decs.append(dec)
    model = chained_AF(
        input_length=data_dim, conditioners=conditioners, softplus=softplus
    )
    return model


def load_f5(filename, float64=False):
    """
    Load in trained f5 modules

    Args:
        filename: str, path to the file
        float64: bool, whether to load the model in float64

    Returns:
        output: the model as a pytree
    """
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        # Initially create the model without converting data types
        model = make_f5(key=jr.PRNGKey(0), **hyperparams)
        # Deserialize the model from the file as is
        output = eqx.tree_deserialise_leaves(f, model)

    return output
