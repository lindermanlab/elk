"""
elk.py
code to write Evaluating Levenberg-Marquardt with Kalman (ELK) code

Note that lambda = 1 / sigmasq. we use lambda in the paper, but sigmasq in the code.
"""

import jax
from jax import vmap
from jax.lax import scan

import jax.numpy as jnp

from dynamax.linear_gaussian_ssm.inference import make_lgssm_params

import dynamax.linear_gaussian_ssm as lgssm

from elk.utils.parallel_kalman_scalar import (
    ScalarParams,
    make_scalar_params,
    parallel_scalar_filter,
)


def elk_alg(
    f,
    initial_state,
    states_guess,
    drivers,
    sigmasq=1e8,
    num_iters=10,
    quasi=False,
    AR=False,
    deer=False,
):
    """
    Run ELK to evaluate the model. Uses a Kalman filter.

    Args:
      f: a forward fxn that takes in a full state and a driver, and outputs the next full state.
          In the context of a GRU, f is a GRU cell, the full state is the hidden state, and the driver is the input
      initial_state: packed_state, jax.Array (DIM,)
      states_guess, jax.Array, (L-1, DIM)
      drivers, jax.Array, (L-1,N_noise)
      sigmasq: float, controls regularization (high sigmasq -> low regularization)
      num_iters: number of iterations to run
      quasi: bool, whether to use quasi-newton or not
      AR: bool, basically evaluate autoregressively (Jacobi iterations, zeroth order) 
      deer: bool, whether to use deer or not (equivalent to sigmasq=infinity, but more numerically stable)
    Notes:
    - The initial_state is NOT the same as the initial mean we give to dynamax
    - The initial_mean is something on which we do inference
    - The initial_state is the fixed starting point.

    The structure looks like the following.
    Let h0 be the initial_state (fixed), h[1:L-1] be the states, and e[0:L-2] be the drivers

    Then our graph looks like

    h0 -----> h1 ---> h2 ---> ..... h_{L-2} ----> h_{L-1}
              |       |                   |          |
              e1      e2       ..... e_{L-2}      e_{L-1}
    """
    DIM = len(initial_state)
    L = len(drivers)

    @jax.vmap
    def full_mat_operator(q_i, q_j):
        """Binary operator for parallel scan of linear recurrence. Assumes a full Jacobian matrix A
        Args:
            q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
            q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
        Returns:
            new element ( A_out, Bu_out )
        """
        A_i, b_i = q_i
        A_j, b_j = q_j
        return A_j @ A_i, A_j @ b_i + b_j

    @jax.vmap
    def diag_mat_operator(q_i, q_j):
        """Binary operator for parallel scan of linear recurrence. Assumes a DIAGONAL Jacobian matrix A
        Args:
            q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
            q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
        Returns:
            new element ( A_out, Bu_out )
        """
        A_i, b_i = q_i
        A_j, b_j = q_j
        return A_j * A_i, A_j * b_i + b_j

    @jax.jit
    def _step(states, args):
        # Evaluate f and its Jacobian in parallel across timesteps 1,..,T-1
        fs = vmap(f)(states[:-1], drivers[1:])
        Jfs = vmap(jax.jacrev(f, argnums=0))(
            states[:-1], drivers[1:]
        )  

        # Compute the As and bs from fs and Jfs
        if quasi:
            As = vmap(lambda Jf: jnp.diag(Jf))(Jfs)
            bs = fs - As * states[:-1]
        elif AR:
            As = jnp.zeros_like(Jfs)
            bs = fs
        else:
            As = Jfs
            bs = fs - jnp.einsum("tij,tj->ti", As, states[:-1])

        if quasi and not deer:
            params = make_scalar_params(
                initial_mean=f(initial_state, drivers[0]),
                dynamics_weights=As,
                dynamics_bias=bs,
                emission_noises=jnp.ones(L) * sigmasq,
            )
        elif deer:
            # initial_state is h0
            b0 = f(initial_state, drivers[0])  # h1
            A0 = jnp.zeros_like(As[0])
            A = jnp.concatenate(
                [A0[jnp.newaxis, :], As]
            )  # (L, D, D) [or (L, D) for quasi]
            b = jnp.concatenate([b0[jnp.newaxis, :], bs])  # (L, D)
            if quasi:
                binary_op = diag_mat_operator
            else:
                binary_op = full_mat_operator
        else:
            params = make_lgssm_params(
                initial_mean=f(initial_state, drivers[0]),
                initial_cov=jnp.eye(DIM),
                dynamics_weights=As,
                dynamics_bias=bs,
                dynamics_cov=jnp.eye(DIM),
                emissions_weights=jnp.eye(DIM),
                emissions_cov=jnp.eye(DIM) * sigmasq,
                emissions_bias=jnp.zeros(DIM),
            )
        # run appropriate parallel alg
        if deer:
            _, new_states = jax.lax.associative_scan(binary_op, (A, b))
        elif quasi:
            post = jax.vmap(
                parallel_scalar_filter,
                in_axes=(
                    ScalarParams(0, 1, 1, None),
                    1,
                ),
                out_axes=1,
            )(params, states)
            new_states = post.filtered_means
        else:
            post = lgssm.parallel_inference.lgssm_filter(params, states)
        new_states = post.filtered_means
        return new_states, new_states

    _, states_iters = scan(_step, states_guess, None, length=num_iters)
    missing_init_state = jnp.vstack((states_guess[None, ...], states_iters))
    everything = jnp.concatenate(
        (
            jnp.broadcast_to(
                initial_state,
                (missing_init_state.shape[0], 1, missing_init_state.shape[-1]),
            ),
            missing_init_state,
        ),
        axis=1,
    )
    return everything
