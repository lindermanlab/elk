"""
fig8.py
script for the lorenz data
A100
"""

import argparse

import jax
import jax.numpy as jnp

from functools import partial

from diffrax import diffeqsolve, SaveAt, ODETerm, Dopri5

import time

import wandb
import pickle

import matplotlib.pyplot as plt

from elk.algs.elk import elk_alg

# global variables
DISCRETE_DT = 0.01
D = 5
F = 8
MAX_STEPS = int(1e9)
L_GEN = 20_000


def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def save_routine(data, name):
    artifact = wandb.Artifact(name, type="dataset")
    save_to_pickle(data, f"{name}.pkl")
    artifact.add_file(f"{name}.pkl")
    wandb.log_artifact(artifact)


def _L96_diffrax(t, y, args):
    """Lorenz 96 model with constant forcing"""
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    d = jax.vmap(lambda i: (y[(i + 1) % D] - y[i - 2]) * y[i - 1] - y[i] + F)(
        jnp.arange(D)
    )
    return d


TERM = ODETerm(_L96_diffrax)
SOLVER = Dopri5()


def f_lorenz96(x_curr, driver):
    """
    Function for ELK
    """
    effective_state = x_curr  # + driver
    solution = diffeqsolve(
        TERM, SOLVER, t0=0, t1=DISCRETE_DT, dt0=DISCRETE_DT, y0=effective_state
    )
    return solution.ys.squeeze()


def step_fn(y, driver):
    new_y = f_lorenz96(y, driver)
    return new_y, new_y


def solution_fn_alt(_y0, _L):
    _, ys = jax.lax.scan(step_fn, _y0, jnp.zeros(_L))
    return ys


def gen_Lorenz96(B, T):
    """
    Generate Lorenz96 data

    Args:
        B: int, number of datasets
        T: int, length of dataset
        discrete_dt: float, time step

    Returns:
        y_BTD (B,T,D): jnp.array, Lorenz96 data
        y0_BD (B,D): jnp.array, initial conditions for sequential
    """
    solution_fn = lambda _y0, _L: diffeqsolve(
        TERM,
        SOLVER,
        t0=0.0,
        t1=(_L + 1) * DISCRETE_DT,
        dt0=DISCRETE_DT,
        y0=_y0,
        saveat=SaveAt(ts=jnp.arange(0.0, (_L + 1) * DISCRETE_DT, step=DISCRETE_DT)),
        max_steps=100_000,
    ).ys[1:]
    y0 = F * jnp.ones(
        D
    )  # Initial state (the actual initial state is mopped up by the driver).
    y0 = y0.at[0].add(0.01)
    y = solution_fn(y0, L_GEN)
    y0_BD = y[-(L_GEN // 2) :: (L_GEN // (2 * B))]
    y_BTD = jax.vmap(solution_fn_alt, in_axes=(0, None))(y0_BD, T)

    y_BTD = y_BTD[:B, :T]

    assert y_BTD.shape == (
        B,
        T,
        D,
    ), f"y_BTD shape is {y_BTD.shape} but\nB={B}\nT={T}\nD={D}"

    return y_BTD, y0_BD


def step_fn(y, driver):
    new_y = f_lorenz96(y, driver)
    return new_y, new_y


def make_lorenz_times_seq(y_BTD, drivers_BTD, nreps=3):
    """
    Time sequential evaluation of the Lorenz96 model

    Output: times (B, nreps, 1)
    """
    B, T, D = y_BTD.shape
    times = jnp.zeros((B, nreps, 1))
    for b in range(B):
        jax.lax.scan(step_fn, jnp.zeros(D), drivers_BTD[b])
        for rep in range(nreps):
            tstart = time.time()
            _ = jax.lax.scan(step_fn, jnp.zeros(D), drivers_BTD[b])
            tend = time.time()
            t = tend - tstart
            times = times.at[b, rep, 0].set(t)
    return times


def step_fn(y, driver):
    new_y = f_lorenz96(y, driver)
    return new_y, new_y


def make_lorenz_tensor(y_BTD, y0_BD, sigmasqs, method, nreps=1):
    """
    Goal is make a tensor with dimensions (B, nreps, sigmasq, newton_iterations)

    Args:
        y_BTD: jnp.array, (B,T,D)
        y0_BD: jnp.array, (B,D), initial condition for sequential scan
        sigmasqs: list of lambdas
        method: 'str' indicating which method ('deer', qdeer', 'elk', 'qelk', 'seq')
        nreps: int, number of repetitions
    """
    B, T, D = y_BTD.shape
    if method == "seq":
        tensor = jnp.zeros((B, nreps, len(sigmasqs)))
    else:
        tensor = jnp.zeros((B, nreps, len(sigmasqs), T))
    times = jnp.zeros((B, nreps, len(sigmasqs)))
    for b in range(B):
        for lidx, sigmasq in enumerate(sigmasqs):
            # jit the functions ahead of time
            closed_deer = jax.jit(
                partial(
                    elk_alg,
                    f_lorenz96,
                    y_BTD[b, 0],
                    jnp.zeros((T - 1, D)),
                    jnp.zeros((T - 1, D)),
                    sigmasq=sigmasq,
                    num_iters=T,
                    deer=True,
                )
            )
            closed_qdeer = jax.jit(
                partial(
                    elk_alg,
                    f_lorenz96,
                    y_BTD[b, 0],
                    jnp.zeros((T - 1, D)),
                    jnp.zeros((T - 1, D)),
                    sigmasq=sigmasq,
                    num_iters=T,
                    deer=True,
                    quasi=True,
                )
            )
            closed_elk = jax.jit(
                partial(
                    elk_alg,
                    f_lorenz96,
                    y_BTD[b, 0],
                    jnp.zeros((T - 1, D)),
                    jnp.zeros((T - 1, D)),
                    sigmasq=sigmasq,
                    num_iters=T,
                )
            )
            closed_qelk = jax.jit(
                partial(
                    elk_alg,
                    f_lorenz96,
                    y_BTD[b, 0],
                    jnp.zeros((T - 1, D)),
                    jnp.zeros((T - 1, D)),
                    sigmasq=sigmasq,
                    num_iters=T,
                    quasi=True,
                )
            )

            def closed_seq():
                _, ys = jax.lax.scan(step_fn, y0_BD[b], jnp.zeros(T))
                return ys

            closed_deer()
            closed_qdeer()
            closed_elk()
            closed_qelk()
            closed_seq()

            method_dict = {
                "deer": closed_deer,
                "qdeer": closed_qdeer,
                "elk": closed_elk,
                "qelk": closed_qelk,
                "seq": closed_seq,
            }

            for rep in range(nreps):
                method_func = method_dict[method]
                tstart = time.time()
                results = method_func()
                tend = time.time()
                t = tend - tstart
                if method == "seq":
                    loss = jnp.sum(jnp.abs(results - y_BTD[b])) / T
                    tensor = tensor.at[b, rep, lidx].set(loss)
                else:
                    losses = (
                        jnp.sum(jnp.abs(results - y_BTD[b]), axis=(1, 2)) / T
                    )  # (T+1,)
                    tensor = tensor.at[b, rep, lidx].set(losses[:-1])
                times = times.at[b, rep, lidx].set(t)  #
    # log summary of times and loss to wandb
    wandb.log({f"{method}-times": jnp.mean(times).item()})
    if method == "seq":
        wandb.log({f"{method}-loss": jnp.mean(tensor).item()})
    else:
        loss_list = jnp.min(jnp.mean(tensor, axis=(0, 1)), axis=-1).tolist()
        wandb.log({f"{method}-loss": loss_list})
    # make some basic plots
    if method != "seq":
        fig, ax = plt.subplots()
        for s in range(len(sigmasqs)):
            ax.plot(
                jnp.mean(tensor[:, :, s], axis=(0, 1)), label=f"sigmasq={sigmasqs[s]}"
            )
        ax.legend()
        ax.set_ylabel("MAD")
        ax.set_xlabel("newton iterations")
        ax.set_title(method)
        ax.set_yscale("log")

        # Log the image to WandB
        wandb.log({f"{method}_error_plot": wandb.Image(fig)})

        # Close the plot
        plt.close(fig)

    return tensor, times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Lorenz data and track performance"
    )
    parser.add_argument("--B", type=int, default=10, help="number of datasets")
    parser.add_argument("--T", type=int, default=1000, help="length of dataset")
    parser.add_argument("--proto", action="store_true", help="smaller prototype run")
    parser.add_argument(
        "--sigmasqs",
        type=float,
        nargs="+",
        default=[3e-1, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 1e5, 1e6],
        help="sigmasqs to test",
    )

    args = parser.parse_args()

    wandb.init(project="elk", entity="linderman-lab")

    B = args.B
    T = args.T
    proto = args.proto
    sigmasqs = args.sigmasqs
    if proto:
        B = 2
        T = 50
        sigmasqs = [1e1, 1e2]
    print(sigmasqs)

    y_BTD, y0_BD = gen_Lorenz96(B, T)

    results_dict = {}
    results_dict["sigmasqs"] = sigmasqs
    results_dict["T"] = T

    for method in ["seq", "qdeer", "deer"]:
        results_dict[method], results_dict[method + "-times"] = make_lorenz_tensor(
            y_BTD, y0_BD, [1], method
        )
        save_routine(results_dict, f"lorenz_hyperparam_tensor_{B}datasets_{T}length")
    for method in ["qelk", "elk"]:
        results_dict[method], results_dict[method + "-times"] = make_lorenz_tensor(
            y_BTD, y0_BD, sigmasqs, method
        )
        save_routine(results_dict, f"lorenz_hyperparam_tensor_{B}datasets_{T}length")

    wandb.finish()
