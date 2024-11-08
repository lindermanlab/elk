"""
fig5.py
quick function for comparing the outputs of quasi and sequential
"""
import jax
import flax
import jax.numpy as jnp

from functools import partial

import matplotlib.pyplot as plt

import wandb
import pickle

import argparse

from elk.algs.deer import seq1d


def compare_outputs(config):
    wandb.init(project="elk", job_type="benchmark", config=config)
    config = wandb.config
    nh = config["nh"]
    nsequence = config["nsequence"]
    seed = config["seed"]
    batch_size = config["batch_size"]
    dtype = jnp.float32

    gru = flax.linen.GRUCell(features=nh, dtype=dtype, param_dtype=dtype)
    key = jax.random.PRNGKey(seed)
    subkey1, subkey2, key = jax.random.split(key, 3)

    carry = gru.initialize_carry(subkey1, (batch_size, nh))  # (batch_size, nh)
    inputs = jax.random.normal(
        subkey2, (nsequence, batch_size, nh), dtype=dtype
    )  # (nsequence, batch_size, nh)
    params = gru.init(key, carry, inputs[0])

    @jax.jit
    def func1(carry: jnp.ndarray, inputs: jnp.ndarray, params) -> jnp.ndarray:
        carry, outputs = jax.lax.scan(partial(gru.apply, params), carry, inputs)
        return outputs

    @jax.jit
    def func_deer(carry: jnp.ndarray, inputs: jnp.ndarray, params) -> jnp.ndarray:

        gru_func = lambda carry, inputs, params: gru.apply(params, carry, inputs)[0]

        output = jax.vmap(
            lambda gru_func, carry, inputs, params: seq1d(
                gru_func, carry, inputs, params
            )[0],
            in_axes=(None, 0, 1, None),
            out_axes=1,
        )(gru_func, carry, inputs, params)

        return output

    @jax.jit
    def func_quasi(carry: jnp.ndarray, inputs: jnp.ndarray, params) -> jnp.ndarray:

        gru_func = lambda carry, inputs, params: gru.apply(params, carry, inputs)[0]

        output = jax.vmap(
            lambda gru_func, carry, inputs, params: seq1d(
                gru_func, carry, inputs, params, quasi=True
            )[0],
            in_axes=(None, 0, 1, None),
            out_axes=1,
        )(gru_func, carry, inputs, params)

        return output

    # compile
    _ = func1(carry, inputs, params)
    _ = func_deer(carry, inputs, params)
    y_seq = func1(carry, inputs, params)[:, 0, 0]  # (nsequence,)
    y_deer = func_deer(carry, inputs, params)[:, 0, 0]  # (nsequence,)
    y_quasi = func_quasi(carry, inputs, params)[:, 0, 0]  # (nsequence,)

    # plotting

    # Configure font sizes.
    SMALL_SIZE = 7 + 3
    MEDIUM_SIZE = 8 + 3
    BIGGER_SIZE = 11 + 3
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Configure the line styles, colors etc.
    cols = {
        "Sequential": "tab:green",
        "DEER": "tab:red",
        "Quasi-DEER": "tab:orange",
    }
    lins = {
        "Sequential": ":",
        "DEER": "-",
        "Quasi-DEER": "-",
    }
    mars = {
        "Sequential": ".",
        "DEER": "+",
        "Quasi-DEER": "x",
    }
    tgs = {
        "Sequential": "seq",
        "DEER": "deer",
        "Quasi-DEER": "quasi",
    }
    labs = [
        "Sequential",
        "DEER",
        "Quasi-DEER",
    ]

    # Create the figure and axis
    fig, ax = plt.subplots(2, 2, figsize=(12, 5))

    x = jnp.arange(nsequence)
    nlast = 200

    # Plot comparison
    ax[0, 0].plot(
        x[-nlast:],
        y_deer[-nlast:],
        label="DEER",
        linestyle=lins["DEER"],
        color=cols["DEER"],
    )
    ax[0, 0].plot(
        x[-nlast:],
        y_seq[-nlast:],
        label="Sequential",
        linestyle=lins["Sequential"],
        color=cols["Sequential"],
    )

    ax[1, 0].plot(
        x[-nlast:],
        y_deer[-nlast:],
        label="Quasi-DEER",
        linestyle=lins["Quasi-DEER"],
        color=cols["Quasi-DEER"],
    )
    ax[1, 0].plot(
        x[-nlast:],
        y_seq[-nlast:],
        label="Sequential",
        linestyle=lins["Sequential"],
        color=cols["Sequential"],
    )

    # Plot difference
    ax[0, 1].plot(y_seq - y_deer)
    ax[0, 1].set_title(
        "Difference between sequential and DEER outputs",
        pad=20,
    )
    ax[0, 1].tick_params(axis="x")
    ax[0, 1].tick_params(axis="y")

    ax[1, 1].plot(y_seq - y_quasi)
    ax[1, 1].set_title(
        "Difference between sequential and quasi-DEER outputs",
        pad=20,
    )
    ax[1, 1].set_xlabel("Sequence index\n(b)")
    ax[1, 1].tick_params(axis="x")
    ax[1, 1].tick_params(axis="y")

    # Setting labels, title, legend, and ticks
    ax[1, 0].set_xlabel("Sequence index\n(a)")
    ax[0, 0].set_title(
        f"GRU outputs for last {nlast} indices, DEER vs Sequential",
        pad=20,
    )
    ax[1, 0].set_title(
        f"GRU outputs for last {nlast} indices, quasi-DEER vs Sequential",
        pad=20,
    )
    ax[0, 0].legend()
    ax[1, 0].legend()
    ax[0, 0].set_xticks(x[-nlast:][::20])  # Set x-ticks for better visualization
    ax[0, 0].set_xticklabels(x[-nlast:][::20])
    ax[0, 0].tick_params(axis="y")

    # Adjust layout for better spacing
    plt.tight_layout(pad=3.0)

    # Save the figure as a .pdf file
    pdf_filename = "fig5.pdf"
    fig.savefig(pdf_filename, format="pdf")

    # Create a wandb artifact and add the .pdf file to it
    artifact_plot = wandb.Artifact("fig5", type="plots")
    artifact_plot.add_file(pdf_filename)

    # Log the artifact to wandb
    wandb.log_artifact(artifact_plot)

    # Close the plot to avoid memory leaks
    plt.close(fig)

    outputs = {"seq": y_seq, "quasi": y_quasi, "deer": y_deer}

    # Save data to a pickle file
    with open("outputs.pkl", "wb") as f:
        pickle.dump(outputs, f)
    # Create a wandb artifact
    artifact_pickle = wandb.Artifact("OutputPickle", type="dataset")

    # Add the pickle file to the artifact
    artifact_pickle.add_file("outputs.pkl")

    # Log the artifact to wandb
    wandb.log_artifact(artifact_pickle)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="compare the outputs of sequential and DEER"
    )
    parser.add_argument("--nh", type=int, default=4, help="dimension of hidden state")
    parser.add_argument(
        "--nsequence", type=int, default=10000, help="length of sequence"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    config = vars(args)
    compare_outputs(config)
