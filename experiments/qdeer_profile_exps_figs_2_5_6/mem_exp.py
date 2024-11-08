"""
mem_exp.py
show memory usage of seq, quasi, and deer
target is to use V100 with 16 GB memory
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import flax.linen
from typing import Any
from functools import partial

import wandb

import matplotlib.pyplot as plt

import sys
import argparse

from benchmark_helper import many_fxn_args_benchmark_memory

from elk.algs.deer import seq1d


def mem_benchmark_seq1d_gru_sweep():
    config = wandb.config
    mem_benchmark_seq1d_gru(config)
    os._exit(0)  # really need this to exit


def mem_benchmark_seq1d_gru(config):

    wandb.init(project="elk", job_type="benchmark", config=config)
    config = wandb.config

    nh = config["nh"]
    dtype = jnp.float32
    seed = config["seed"]
    batch_size = config["batch_size"]
    nsequence = config["nsequence"]
    alg = config["alg"]
    WITH_JIT = config["with_jit"]
    NWARMUPS = config["nwarmups"]
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
            return outputs

    elif alg == "quasi":
        # mem-efficient quasi
        def func(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any) -> jnp.ndarray:

            gru_func = lambda carry, inputs, params: gru.apply(params, carry, inputs)[0]

            output = jax.vmap(
                lambda gru_func, carry, inputs, params: seq1d(
                    gru_func, carry, inputs, params, quasi=True
                )[0],
                in_axes=(None, 0, 1, None),
                out_axes=1,
            )(gru_func, carry, inputs, params)

            return output

    elif alg == "deer":
        # full deer
        def func(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any) -> jnp.ndarray:

            gru_func = lambda carry, inputs, params: gru.apply(params, carry, inputs)[0]
            output = jax.vmap(
                lambda gru_func, carry, inputs, params: seq1d(
                    gru_func, carry, inputs, params
                )[0],
                in_axes=(None, 0, 1, None),
                out_axes=1,
            )(gru_func, carry, inputs, params)

            return output

    else:
        raise ValueError("alg must be one of seq, quasi, or deer")

    fxn_arg_dict = {
        "mem": {
            "func": func,
            "args": (carry, inputs, params),
        },
    }

    results = many_fxn_args_benchmark_memory(
        fxn_arg_dict, with_jit=WITH_JIT, nwarmups=NWARMUPS, nreps=nreps
    )
    wandb.log(results)
    wandb.finish()
    os._exit(0)  # really need os._exit() for the multithreading


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="profile mem on GRU benchmark")

    parser.add_argument("--nh", type=int, default=64, help="hidden size")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--nsequence", type=int, default=30_000, help="sequence length")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--alg", type=str, default="seq", help="which of seq, quasi, or deer to run"
    )
    parser.add_argument(
        "--with_jit",
        type=bool,
        default=True,
        help="whether to JIT compile the functions",
    )
    parser.add_argument(
        "--nwarmups", type=int, default=0, help="number of warmup iterations to perform"
    )
    parser.add_argument("--nreps", type=int, default=1, help="number of reps to run")

    # Argument for WandB sweep
    parser.add_argument(
        "--sweep_id", type=str, help="the wandb sweep id to use for a sweep"
    )

    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = "linderman-lab/elk/" + args.sweep_id
        wandb.agent(sweep_id=sweep_id, count=1, function=mem_benchmark_seq1d_gru_sweep)
    else:
        config = {
            "nh": args.nh,
            "batch_size": args.batch_size,
            "nsequence": args.nsequence,
            "seed": args.seed,
            "alg": args.alg,
            "with_jit": args.with_jit,
            "nwarmups": args.nwarmups,
            "nreps": args.nreps,
        }
        mem_benchmark_seq1d_gru(config)
