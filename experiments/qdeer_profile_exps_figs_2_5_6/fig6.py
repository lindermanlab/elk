"""
fig6.py
this python file is meant to generate the plot for figure 6 (supplement in the neurips submission)
this plot is of timing and memory consumption on a 32GB V100 with batch size of 1

the memory wandb sweep: "xavier_gonzalez/ike/v3r9en3k"
"""

import pickle
import wandb
from tqdm import tqdm
import pandas as pd
import numpy as np

# time_pickle = "/Users/xaviergonzalez/Desktop/xavier_folders/stanford/linderman/f5/ifl-equinox/code/notebooks/xavier-nbs/deerep_batchsize1_GB32_stamp_19_21_25.pickle"
# # memsweep_string = "xavier_gonzalez/ike/v3r9en3k"
# memsweep_string = "xavier_gonzalez/ike/t0f5qpao"

time_pickle = "fig6_time_20seeds_bs1_V100_32G_0721.pickle"
mem_pickle = "fig6_mem_3seeds_bs1_V100_32G_0721.pickle"

with open(time_pickle, "rb") as f:
    time_df = pickle.load(f)

with open(mem_pickle, "rb") as f:
    mem_df = pickle.load(f)

# # load memory results
# api = wandb.Api(timeout=29)
# mem_sweep = api.sweep(memsweep_string)

# mem_sweep_data = []
# for run in tqdm(mem_sweep.runs):
#     run_data = run.history()
#     run_data["run_id"] = run.id
#     for key, value in run.config.items():
#         run_data[key] = value
#     mem_sweep_data.append(run_data)

# mem_df = pd.concat(mem_sweep_data, ignore_index=True)

# configure plotting

# Use TeX.
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

# Configure font sizes.
SMALL_SIZE = 7 + 1
MEDIUM_SIZE = 8 + 1
BIGGER_SIZE = 10 + 1
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Configure the line styles, colors etc.
cols = {
    "Sequential": "tab:green",
    "DEER": "tab:red",
    "Quasi-DEER": "tab:orange",
    "IKE": "tab:blue",
    "Quasi-IKE": "tab:purple",
}
lins = {"Sequential": ":", "DEER": "-", "Quasi-DEER": "-", "IKE": "-", "Quasi-IKE": "-"}
mars = {"Sequential": ".", "DEER": "+", "Quasi-DEER": "x", "IKE": "+", "Quasi-IKE": "x"}
tgs = {
    "Sequential": "seq",
    "DEER": "deer",
    "Quasi-DEER": "quasi",
    "IKE": "ike",
    "Quasi-IKE": "qike",
}
labs = [
    "Sequential",
    "DEER",
    "Quasi-DEER",
]  # We dont run these here: ['IKE', 'Quasi-IKE']

# These are the configs we will plot.
ls = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
hs = [1, 2, 4, 8, 16, 32, 64]


def make_fig_6(time_df, mem_df):
    fig, axes = plt.subplots(2, 7, sharex=True, figsize=(7, 5))

    for lab in ["Sequential", "DEER", "Quasi-DEER"]:

        for i, h in enumerate(hs):
            try:
                time_uqs, time_mds, time_lqs = [], [], []
                for l in ls:
                    y = time_df[
                        (time_df["nsequence"] == l)
                        & (time_df["nh"] == h)
                        & (time_df["alg"] == tgs[lab])
                    ]["time"].values
                    time_uqs.append(np.nanquantile(y, 0.75))
                    time_mds.append(np.nanquantile(y, 0.50))
                    time_lqs.append(np.nanquantile(y, 0.25))
            except Exception as e:
                print("Time failure")
                print(e)

            # analyze memory
            try:
                mem_uqs, mem_mds, mem_lqs = [], [], []
                for l in ls:
                    y = (
                        mem_df[
                            (mem_df["nsequence"] == l)
                            & (mem_df["nh"] == h)
                            & (mem_df["alg"] == tgs[lab])
                        ]["mem"].values
                        / 1000
                    )  # convert to GB
                    mem_uqs.append(np.nanquantile(y, 0.75))
                    mem_mds.append(np.nanquantile(y, 0.50))
                    mem_lqs.append(np.nanquantile(y, 0.25))
            except Exception as e:
                print("Mem failure")
                print(e)

            # plot time
            if any(np.isfinite(time_mds)):
                axes[0, i].plot(
                    ls[: len(time_mds)],
                    time_mds,
                    c=cols[lab],
                    marker=mars[lab],
                    linestyle=lins[lab],
                    markersize=4,
                    linewidth=0.5,
                )

            # plot memory
            if any(np.isfinite(mem_mds)):
                axes[1, i].plot(
                    ls[: len(mem_mds)],
                    mem_mds,
                    c=cols[lab],
                    marker=mars[lab],
                    linestyle=lins[lab],
                    markersize=4,
                    linewidth=0.5,
                )

    # All axis.
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)

    # Set y-tick labels for timing
    yticks = [1e-4, 1e-2, 1]
    yticklabels = ["$10^{-4}$", "$10^{-2}$", "$10^0$"]
    for ax in axes[0]:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    # set y-tick lablels for mem
    yticks_mem = [1, 10]
    yticklabels_mem = ["$10^{0}$", "$10^{1}$"]
    for ax in axes[1]:
        ax.axhline(32, linestyle="--", c="k")  # set dashed line for mem cap
        ax.set_yticks(yticks_mem)
        ax.set_yticklabels(yticklabels_mem)

    # remove y tick lables
    for ax in axes[0, 1:]:
        ax.set_yticklabels([])

    for ax in axes[1, 1:]:
        ax.set_yticklabels([])

    # Top axis.
    for i, ax in enumerate(axes[0]):
        # ax.set_ylim(0.0001, 0.01)  # 10.0)
        ax.set_title(f"$D = {hs[i]}$")
        # ax.axhline(16, linestyle="--", c="k")
        ax.set_xticks(
            ls,
            ["1K", "3K", "10K", "30K", "100K", "300K", "1M"],
            rotation=90,
            ha="center",
            va="top",
        )
        ax.set_xlim((min(ls) * 0.9, max(ls) * 1.1))
        ax.set_ylim(1e-4, 20)

    # Bottom axis
    for i, ax in enumerate(axes[1]):
        ax.set_xticks(
            ls,
            ["1K", "3K", "10K", "30K", "100K", "300K", "1M"],
            rotation=90,
            ha="center",
            va="top",
        )
        ax.set_xlim((min(ls) * 0.9, max(ls) * 1.1))
        ax.set_ylim(0.5, 40)

    # Adjust the position of the x suplabel.
    t = fig.supxlabel(f"Sequence Length (T)", fontsize=MEDIUM_SIZE)
    t.set_position(
        (
            t.get_position()[0],
            t.get_position()[1] - 0.3,
        )
    )

    # Set each y-label up.
    axes[0, 0].set_ylabel("Wallclock (s)")
    axes[1, 0].set_ylabel("Memory (GB)", labelpad=9)

    # Now fake the legend.
    ax = axes[0, 1]
    for lab in labs:
        ax.plot(
            [-1, -1],
            [-1, -1],
            c=cols[lab],
            marker=mars[lab],
            linestyle=lins[lab],
            label=lab,
        )
    leg = ax.legend(
        ncols=3,
        handletextpad=0.5,
        bbox_to_anchor=(4.8, 1.25),
        bbox_transform=ax.transAxes,
    )

    # Format and save
    plt.tight_layout()
    fig.subplots_adjust(top=0.82, bottom=-0.15, right=0.98)  # Adjusted bottom and top
    plt.savefig(
        "./figure_timing_supp.pdf",
        bbox_inches="tight",
        pad_inches=0.01,
    )


if __name__ == "__main__":
    make_fig_6(time_df, mem_df)
