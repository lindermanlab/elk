"""
fig8_bottom.py
script to record some traces from lorenz96
"""

import argparse

import jax
import jax.numpy as jnp

from functools import partial

from diffrax import diffeqsolve, SaveAt, ODETerm, Dopri5

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
    Function for elk
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Lorenz data and track performance"
    )
    parser.add_argument("--proto", action="store_true", help="smaller prototype run")
    parser.add_argument("--SEED", type=int, default=7, help="which dataset to use")
    parser.add_argument("--elk_sigmsq", type=float, default=3e2, help="sigmasq for elk")
    parser.add_argument(
        "--qelk_sigmsq", type=float, default=1e1, help="sigmasq for qelk"
    )

    args = parser.parse_args()

    wandb.init(project="elk", entity="xavier_gonzalez")

    B = 10
    T = 1000
    D = 5
    SEED = args.SEED
    proto = args.proto
    if proto:
        T = 50

    y_BTD, y0_BD = gen_Lorenz96(B, T)

    y = y_BTD[SEED]

    sigmasq_dict = {
        "deer": 1.0,
        "qdeer": 1.0,
        "elk": args.elk_sigmsq,
        "qelk": args.qelk_sigmsq,
    }

    results = {}
    results["true"] = y
    for method in ["qdeer", "deer", "qelk", "elk"]:
        deer_flag = method in ["deer", "qdeer"]
        quasi_flag = method in ["qdeer", "qelk"]
        results[method] = elk_alg(
            f_lorenz96,
            y[0],
            jnp.zeros((T - 1, D)),
            jnp.zeros((T - 1, D)),
            sigmasq=sigmasq_dict[method],
            num_iters=(T // 2),
            deer=deer_flag,
            quasi=quasi_flag,
        )
        save_routine(results, f"lorenz_traces_B_{B}_T_{T}_seed_{SEED}")

    # log the plot to wandb for quick verification of accuracy
    fig, ax = plt.subplots()
    for method in ["qdeer", "deer", "qelk", "elk"]:
        ax.plot(jnp.sum(jnp.abs(results[method] - y), axis=(1, 2)) / T, label=method)
    ax.legend()
    ax.set_ylabel("MAD")
    ax.set_xlabel("newton iterations")
    ax.set_title(method)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e4)
    ax.legend()

    # Log the image to WandB
    wandb.log({"error_plot": wandb.Image(fig)})

    # Close the plot
    plt.close(fig)

    wandb.finish()
