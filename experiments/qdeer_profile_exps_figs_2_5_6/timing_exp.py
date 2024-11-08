"""
gru_timing_exp.py
timing experiments for GRU
In order to generate Figures2 and Figures6, the preferred approach is to set up a sweep in wandb, and to pass the sweep id as an argument (eg in fig2_time.sh)
"""

import jax
import jax.numpy as jnp
import flax.linen
from typing import Any
from functools import partial

import wandb

import matplotlib.pyplot as plt


import sys
import os

import argparse

from benchmark_helper import many_fxn_args_benchmark_timing
from elk.algs.deer import seq1d


def benchmark_seq1d_gru_sweep():
    config = wandb.config
    benchmark_seq1d_gru(config)


def benchmark_seq1d_gru(config):

    wandb.init(project="elk", job_type="benchmark", config=config)
    config = wandb.config

    nh = config["nh"]
    dtype = jnp.float32
    seed = config["seed"]
    batch_size = config["batch_size"]
    nsequence = config["nsequence"]
    alg = config["alg"]
    nreps = config["nreps"]

    gru = flax.linen.GRUCell(features=nh, dtype=dtype, param_dtype=dtype)
    key = jax.random.PRNGKey(seed)
    subkey1, subkey2, key = jax.random.split(key, 3)

    carry = gru.initialize_carry(subkey1, (batch_size, nh))  # (batch_size, nh)
    inputs = jax.random.normal(
        subkey2, (nsequence, batch_size, nh), dtype=dtype
    )  # (nsequence, batch_size, nh)
    params = gru.init(key, carry, inputs[0])

    if alg == "seq":
        # sequential
        def func(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any) -> jnp.ndarray:
            carry, outputs = jax.lax.scan(partial(gru.apply, params), carry, inputs)
            return outputs, 1

    elif alg == "quasi":
        # mem-efficient quasi
        def func(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any) -> jnp.ndarray:

            gru_func = lambda carry, inputs, params: gru.apply(params, carry, inputs)[0]

            output = jax.vmap(
                lambda gru_func, carry, inputs, params: seq1d(
                    gru_func, carry, inputs, params, quasi=True
                ),
                in_axes=(None, 0, 1, None),
            )(gru_func, carry, inputs, params)

            return output

    elif alg == "deer":
        # full deer
        def func(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any) -> jnp.ndarray:

            gru_func = lambda carry, inputs, params: gru.apply(params, carry, inputs)[0]
            output = jax.vmap(
                lambda gru_func, carry, inputs, params: seq1d(
                    gru_func, carry, inputs, params
                ),
                in_axes=(None, 0, 1, None),
            )(gru_func, carry, inputs, params)

            return output

    else:
        raise ValueError("alg must be one of seq, quasi, or deer")


    fxn_arg_dict = {
        "time": {
            "func": func,
            "args": (carry, inputs, params),
        },
    }

    results = many_fxn_args_benchmark_timing(
        fxn_arg_dict, with_jit=True, nwarmups=5, nreps=nreps
    )


    wandb.log(results)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="time the GRU benchmark")

    parser.add_argument("--nh", type=int, default=4, help="hidden size")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--nsequence", type=int, default=100, help="sequence length")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--nreps", type=int, default=2, help="number of reps to run")
    parser.add_argument("--alg", type=str, default="seq", help="seq, deer, or quasi")

    # Argument for WandB sweep
    parser.add_argument(
        "--sweep_id", type=str, help="the wandb sweep id to use for a sweep"
    )

    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = "linderman-lab/elk/" + args.sweep_id
        wandb.agent(sweep_id=sweep_id, count=1, function=benchmark_seq1d_gru_sweep)
    else:
        config = vars(args)
        benchmark_seq1d_gru(config)
