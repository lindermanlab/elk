"""
fig7.py
A100
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial

import os
import sys

import pickle
import wandb

import argparse

import time

from tqdm import tqdm

from elk.algs.deer import seq1d
from elk.algs.elk import elk_alg
from elk.utils.elk_utils import make_unpacking_tuple, packed_single_step, pack_state
from elk.models.flow import load_f5
from elk.models.model_utils import sample_AR_many_layers


def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def save_routine(data, name):
    artifact = wandb.Artifact(name, type="dataset")
    save_to_pickle(data, f"{name}.pkl")
    artifact.add_file(f"{name}.pkl")
    wandb.log_artifact(artifact)


def make_argru_tensor(seeds, sigmasqs, method, nreps=1, T=10_000):
    """
    Goal is to make a tensor with dimensions (seeds, nreps, sigmasq, newton_iterations)
    We should track the residual at each point

    Args:
        seeds: list of seeds
        sigmasqs: list of lambdas
        method: 'str' indicating which method
        nreps: int, number of repetitions
        T: int, sequence length
    """
    tensor = jnp.zeros((len(seeds), nreps, len(sigmasqs), T))
    times = jnp.zeros((len(seeds), nreps, len(sigmasqs)))
    for sidx, seed in tqdm(enumerate(seeds)):
        # set up this seed
        n_layers = 1
        softplus = True
        hparams = {
            "data_dim": T,
            "model_Ps": [3],
            "stack_length": 1,  # 3
            "n_layers": n_layers,  # 2
            "dec_scale": 0.1,
            "skip": True,
            "use_D": False,
            "softplus": softplus,
            "ssm_type": "rnn",
        }
        argru_trained = load_f5("argru_sine_wave_L=10000.eqx")

        N_LAYERS = argru_trained.conditioners[0].seq_model.n_layers
        P = argru_trained.conditioners[0].seq_model.ssm.P  # state size for SSM
        H = argru_trained.conditioners[
            0
        ].seq_model.ssm.H  # state size for input and output
        DIM = N_LAYERS * P + 1
        KEY = jr.split(jr.PRNGKey(seed), 1)[0]
        AR, *_, noises, init_state = sample_AR_many_layers(
            argru_trained, T, key=KEY, dtype=jnp.float32
        )
        # arguments to feed into IKE
        unpacking_tuple = make_unpacking_tuple(argru_trained)
        fxn = partial(packed_single_step, argru_trained, unpacking_tuple)
        initial_packed_state = pack_state(init_state)
        states_guess = jnp.zeros((T - 1, DIM))
        drivers = noises
        for lidx, sigmasq in enumerate(sigmasqs):

            # jit the functions ahead of time

            def fxn_for_deer(state, driver, params):
                return packed_single_step(argru_trained, unpacking_tuple, state, driver)

            closed_deer = jax.jit(
                partial(
                    seq1d,
                    fxn_for_deer,
                    initial_packed_state,
                    drivers[1:],
                    None,
                    max_iter=T,
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
                    max_iter=T,
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
                    sigmasq=sigmasq,
                    num_iters=T,
                )
            )
            closed_qelk = jax.jit(
                partial(
                    elk_alg,
                    fxn,
                    initial_packed_state,
                    states_guess,
                    drivers[1:],
                    sigmasq=sigmasq,
                    num_iters=T,
                    quasi=True,
                )
            )

            closed_deer()
            closed_qdeer()
            closed_elk()
            closed_qelk()

            for rep in range(nreps):
                if method == "deer":
                    tstart = time.time()
                    results, *_ = closed_deer()
                    tend = time.time()
                    t = tend - tstart
                    results = jax.vmap(
                        lambda seq: jnp.concatenate(
                            (initial_packed_state.reshape(1, -1), seq), axis=0
                        )
                    )(results)
                elif method == "qdeer":
                    tstart = time.time()
                    results, *_ = closed_qdeer()
                    tend = time.time()
                    t = tend - tstart
                    results = jax.vmap(
                        lambda seq: jnp.concatenate(
                            (initial_packed_state.reshape(1, -1), seq), axis=0
                        )
                    )(results)
                elif method == "elk":
                    tstart = time.time()
                    results = closed_elk()
                    tend = time.time()
                    t = tend - tstart
                elif method == "qelk":
                    tstart = time.time()
                    results = closed_qelk()
                    tend = time.time()
                    t = tend - tstart
                if jnp.isnan(results).any():
                    print("there are some nans")
                losses = jnp.sum(jnp.abs(results[..., 0] - AR.flatten()) / T, axis=1)
                tensor = tensor.at[sidx, rep, lidx].set(losses[:-1])
                times = times.at[sidx, rep, lidx].set(t)
    return tensor, times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument("--proto", action="store_true", help="smaller prototype run")
    parser.add_argument("--float64", action="store_true", help="use float64")
    parser.add_argument("--n_seeds", type=int, default=15, help="number of seeds")
    parser.add_argument(
        "--sigmasqs",
        type=float,
        nargs="+",
        default=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 3e7, 1e8, 3e8],
        help="sigmasqs to test",
    )
    args = parser.parse_args()

    wandb.init(project="elk", entity="linderman-lab")
    n_seeds = args.n_seeds
    if args.float64:
        jax.config.update("jax_enable_x64", True)
    if args.proto:
        seeds = [0, 1]
        n_seeds = len(seeds)
        sigmasqs = [1e2, 1e3, 1e4, 1e5]
        T = 100
    else:
        seeds = list(range(n_seeds))
        sigmasqs = args.sigmasqs
        T = 10_000
    methods = ["qelk", "elk"]
    print(sigmasqs)

    results_dict = {}
    results_dict["seeds"] = seeds
    results_dict["sigmasqs"] = sigmasqs
    results_dict["T"] = T
    save_routine(results_dict, f"AR_GRU_hyperparam_tensor_{n_seeds}seeds")

    results_dict["qdeer"], results_dict["qdeer-times"] = make_argru_tensor(
        seeds, [1], "qdeer", T=T
    )
    save_routine(results_dict, f"AR_GRU_hyperparam_tensor_{n_seeds}seeds")
    results_dict["deer"], results_dict["deer-times"] = make_argru_tensor(
        seeds, [1], "deer", T=T
    )
    save_routine(results_dict, f"AR_GRU_hyperparam_tensor_{n_seeds}seeds")
    for method in methods:
        results_dict[method], results_dict[method + "-times"] = make_argru_tensor(
            seeds, sigmasqs, method, T=T
        )
        save_routine(results_dict, f"AR_GRU_hyperparam_tensor_{n_seeds}seeds")

    wandb.finish()
