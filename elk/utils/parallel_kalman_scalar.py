"""
parallel_kalman_scalar.py
code for parallel kalman filtering and smoothing on a scalar system
adapted from dynamax/linear_gaussian_ssm/parallel_inference.py
https://github.com/probml/dynamax/blob/main/dynamax/linear_gaussian_ssm/parallel_inference.py


Dynamax

      F₀,Q₀           F₁,Q₁         F₂,Q₂
Z₀ ─────────── Z₁ ─────────── Z₂ ─────────── Z₃ ─────...
|              |              |              |
| H₀,R₀        | H₁,R₁        | H₂,R₂        | H₃,R₃
|              |              |              |
Y₀             Y₁             Y₂             Y₃ 
"""

import jax.numpy as jnp
from jax import lax, vmap

from jaxtyping import Array, Float
from typing import NamedTuple, Union, Optional
from functools import partial


class ScalarParams(NamedTuple):
    """
    Class to hold the params we are going to use in ELK
    We hold for both the sequence length and the state dimension
    """

    initial_mean: Float[Array, "state_dim"]
    dynamics_weights: Float[Array, "ntime state_dim"]
    dynamics_bias: Float[Array, "ntime state_dim"]
    emission_noises: Float[Array, "ntime state_dim"]


def make_scalar_params(initial_mean, dynamics_weights, dynamics_bias, emission_noises):
    return ScalarParams(initial_mean, dynamics_weights, dynamics_bias, emission_noises)


# ---------------------------------------------------------------------------#
#                                Filtering                                  #
# ---------------------------------------------------------------------------#


class PosteriorScalarFilter(NamedTuple):
    r"""Marginals of the Gaussian filtering posterior.
    :param filtered_means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$
    """

    filtered_means: Optional[Float[Array, "ntime"]] = None
    filtered_covariances: Optional[Float[Array, "ntime"]] = None


class FilterMessageScalar(NamedTuple):
    """
    Filtering associative scan elements.

    Note that every one of these is a scalar in our formulation
    but tha they span the sequence length

    Attributes:
        A: P(z_j | y_{i:j}, z_{i-1}) weights.
        b: P(z_j | y_{i:j}, z_{i-1}) bias.
        C: P(z_j | y_{i:j}, z_{i-1}) covariance.
        J:   P(z_{i-1} | y_{i:j}) covariance.
        eta: P(z_{i-1} | y_{i:j}) mean.
    """

    A: Float[Array, "ntime"]
    b: Float[Array, "ntime"]
    C: Float[Array, "ntime"]
    J: Float[Array, "ntime"]
    eta: Float[Array, "ntime"]


def _initialize_filtering_messages(params, emissions):
    """Preprocess observations to construct input for filtering assocative scan."""

    num_timesteps = emissions.shape[0]

    def _first_message(params, y):
        m = params.initial_mean
        sigma2 = params.emission_noises[0]

        S = jnp.ones(1) + sigma2

        A = jnp.zeros(1)
        b = m + (y - m) / S
        C = jnp.ones(1) - (S**-1)
        eta = jnp.zeros(1)
        J = jnp.ones(1)

        return A, b, C, J, eta

    @partial(vmap, in_axes=(None, 0, 0))
    def _generic_message(params, y, t):
        """
        Notes:
            * y are the observations
            * note that the dynamics params and the emissions params are shifted by 1 in a sense
        """
        F = params.dynamics_weights[t]
        b = params.dynamics_bias[t]
        sigma2 = params.emission_noises[t + 1]  # shift

        K = 1 / (1 + sigma2)

        eta = F * K * (y - b)
        J = (F**2) * K

        A = F - K * F
        b = b + K * (y - b)
        C = 1 - K

        return A, b, C, J, eta

    A0, b0, C0, J0, eta0 = _first_message(params, emissions[0])
    At, bt, Ct, Jt, etat = _generic_message(
        params,
        emissions[1:],
        jnp.arange(
            len(emissions) - 1
        ),  # the dynamics params at step 0 generated step 1
    )

    return FilterMessageScalar(
        A=jnp.concatenate([A0, At]),
        b=jnp.concatenate([b0, bt]),
        C=jnp.concatenate([C0, Ct]),
        J=jnp.concatenate([J0, Jt]),
        eta=jnp.concatenate([eta0, etat]),
    )


def parallel_scalar_filter(
    params: ScalarParams,
    emissions: Float[Array, "ntime"],
):
    """
    Notes:
        * really meant to solve the scalar problem
    """

    @vmap
    def _operator(elem1, elem2):
        A1, b1, C1, J1, eta1 = elem1
        A2, b2, C2, J2, eta2 = elem2

        denom = C1 * J2 + 1

        A = (A1 * A2) / denom
        b = A2 * (C1 * eta2 + b1) / denom + b2
        C = C1 * (A2**2) / denom + C2

        # eta = A1 * J2 * (eta2 / J2 - b1) / denom + eta1 # not stable numerically
        eta = A1 * (eta2 - J2 * b1) / denom + eta1
        J = J2 * (A1**2) / denom + J1

        return FilterMessageScalar(A, b, C, J, eta)

    initial_messages = _initialize_filtering_messages(params, emissions)
    final_messages = lax.associative_scan(_operator, initial_messages)

    return PosteriorScalarFilter(
        filtered_means=final_messages.b,
        filtered_covariances=final_messages.C,
    )