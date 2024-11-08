"""
argru_exp.py
Aim to run on an A100
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import wandb

from functools import partial

import argparse

import time

import matplotlib.pyplot as plt
import pickle
import os
import sys

from elk.algs.deer import seq1d
from elk.algs.elk import elk_alg
from elk.utils.elk_utils import make_unpacking_tuple, packed_single_step, pack_state
from elk.models.flow import load_f5
from elk.models.model_utils import sample_AR_many_layers


def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def compare_stopping_time_sweep():
    config = wandb.config
    compare_stopping_time(config)


def compare_stopping_time(config):
    """
    Compare the stopping time of DEER and ELK on an AR GRU
    """
    # get set up
    wandb.init(project="elk", job_type="benchmark", config=config)
    config = wandb.config

    seed = config["seed"]
    L = config["L"]
    nreps = config["nreps"]

    max_iter = L
    elk_sigmasq = 1e6
    qelk_sigmasq = 1e2

    model = load_f5(f"argru_sine_wave_L=10000.eqx")
    N_LAYERS = model.conditioners[0].seq_model.n_layers
    P = model.conditioners[0].seq_model.ssm.P  # state size for SSM
    H = model.conditioners[0].seq_model.ssm.H  # state size for input and output
    DIM = N_LAYERS * P + 1

    # seed specific
    key = jr.split(jr.PRNGKey(seed), 1)[0]

    # jit everything ahead of time
    xs_AR_star, *_, noises, init_state = sample_AR_many_layers(
        model, L, key=key, dtype=jnp.float32
    )
    unpacking_tuple = make_unpacking_tuple(model)
    fxn = partial(packed_single_step, model, unpacking_tuple)
    initial_packed_state = pack_state(init_state)
    states_guess = jnp.zeros((L - 1, DIM))
    drivers = noises

    def fxn_for_deer(state, driver, params):
        return packed_single_step(model, unpacking_tuple, state, driver)

    closed_ar = jax.jit(
        partial(sample_AR_many_layers, model, L, key=key, dtype=jnp.float32)
    )
    closed_deer = jax.jit(
        partial(
            seq1d,
            fxn_for_deer,
            initial_packed_state,
            drivers[1:],
            None,
            max_iter=max_iter,
            full_trace=True,
        )
    )
    closed_qdeer = jax.jit(
        partial(
            seq1d,
            fxn_for_deer,
            initial_packed_state,
            drivers[1:],
            None,
            max_iter=max_iter,
            quasi=True,
            qmem_efficient=False,
            full_trace=True,
        )
    )
    closed_elk = jax.jit(
        partial(
            elk_alg,
            fxn,
            initial_packed_state,
            states_guess,
            drivers[1:],
            sigmasq=elk_sigmasq,
            num_iters=max_iter,
        )
    )
    closed_qelk = jax.jit(
        partial(
            elk_alg,
            fxn,
            initial_packed_state,
            states_guess,
            drivers[1:],
            sigmasq=qelk_sigmasq,
            num_iters=max_iter,
            quasi=True,
        )
    )

    # run the functions to jit them
    closed_ar()
    closed_deer()
    closed_qdeer()
    closed_elk()
    closed_qelk()

    for _ in range(nreps):
        # AR / setup
        t1 = time.time()
        closed_ar()
        t2 = time.time()
        ar_time = t2 - t1  # note that we don't divide AR by max_iters

        # deer
        t3 = time.time()
        deer, *_ = closed_deer()
        t4 = time.time()
        deer_time = (t4 - t3) / max_iter

        # qdeer
        t5 = time.time()
        qdeer, *_ = closed_qdeer()
        t6 = time.time()
        qdeer_time = (t6 - t5) / max_iter

        # elk
        t7 = time.time()
        elk = closed_elk()
        t8 = time.time()
        elk_time = (t8 - t7) / max_iter

        # qelk
        t9 = time.time()
        qelk = closed_qelk()
        t10 = time.time()
        qelk_time = (t10 - t9) / max_iter

        # log
        results = {
            "ar_time": ar_time,
            "deer_time": deer_time,
            "qdeer_time": qdeer_time,
            "elk_time": elk_time,
            "qelk_time": qelk_time,
        }
        wandb.log(results)

    # clean up the deer traces
    deer = jax.vmap(
        lambda seq: jnp.concatenate((initial_packed_state.reshape(1, -1), seq), axis=0)
    )(deer)
    qdeer = jax.vmap(
        lambda seq: jnp.concatenate((initial_packed_state.reshape(1, -1), seq), axis=0)
    )(qdeer)

    # mean absolute deviation
    elk_losses = jnp.mean(jnp.abs(elk[..., 0] - xs_AR_star.flatten()), axis=1)
    qelk_losses = jnp.mean(jnp.abs(qelk[..., 0] - xs_AR_star.flatten()), axis=1)
    deer_losses = jnp.mean(jnp.abs(deer[..., 0] - xs_AR_star.flatten()), axis=1)
    qdeer_losses = jnp.mean(jnp.abs(qdeer[..., 0] - xs_AR_star.flatten()), axis=1)

    losses = {
        "deer_loss": jnp.min(deer_losses),
        "qdeer_loss": jnp.min(qdeer_losses),
        "elk_loss": jnp.min(elk_losses),
        "qelk_loss": jnp.min(qelk_losses),
    }
    wandb.log(losses)

    # save the pickles
    print(f"nans in AR? {jnp.any(jnp.isnan(xs_AR_star))}")
    print(f"nans in DEER? {jnp.any(jnp.isnan(deer))}")
    print(f"nans in qDEER? {jnp.any(jnp.isnan(qdeer))}")
    print(f"nans in ELK? {jnp.any(jnp.isnan(elk))}")
    print(f"nans in qELK? {jnp.any(jnp.isnan(qelk))}")

    results_dict = {
        "AR": xs_AR_star,
        "DEER": deer,
        "qDEER": qdeer,
        "ELK": elk,
        "qELK": qelk,
    }

    artifact = wandb.Artifact(f"results_seed_{seed}_L_{L}", type="dataset")
    save_to_pickle(results_dict, f"results_seed_{seed}_L_{L}.pkl")
    artifact.add_file(f"results_seed_{seed}_L_{L}.pkl")
    wandb.log_artifact(artifact)

    # Plot errors
    fig, ax = plt.subplots()

    ax.plot(deer_losses, label="DEER")
    ax.plot(qdeer_losses, label="qDEER")
    ax.plot(elk_losses, label=f"ELK sigmsq is 1e{round(jnp.log10(elk_sigmasq))}")
    ax.plot(qelk_losses, label=f"qELK sigmsq is 1e{round(jnp.log10(qelk_sigmasq))}")

    ax.set_ylabel("MAD")
    ax.set_xlabel("newton iterations")
    ax.set_title(f"seed is {seed}, L={L}")
    ax.set_yscale("log")
    ax.legend()

    # Log the image to WandB
    wandb.log({"error_plot": wandb.Image(fig)})

    # Close the plot
    plt.close(fig)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="loss plots for length 10_000 sine wave AR GRU"
    )
    parser.add_argument("--seed", type=int, default=305, help="random seed")
    parser.add_argument("--L", type=int, default=10000, help="length of sine wave")
    parser.add_argument("--nreps", type=int, default=4, help="number of repetitions")

    # Argument for WandB sweep
    parser.add_argument(
        "--sweep_id", type=str, help="the wandb sweep id to use for a sweep"
    )

    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = "linderman-lab/elk/" + args.sweep_id
        wandb.agent(sweep_id=sweep_id, function=compare_stopping_time_sweep)
    else:
        config = vars(args)
        compare_stopping_time(config)
