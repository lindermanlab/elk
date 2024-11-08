"""
fig2.py
plotting for figure 2
the experiments are run by fig2_time.sh and fig2_mem.sh, which call timing_exp.py and mem_exp.py
"""

import pickle
import wandb
from tqdm import tqdm
import pandas as pd
import numpy as np

time_pickle = "fig2_time_l95kx626.pkl"
with open(time_pickle, "rb") as f:
    time_df = pickle.load(f)

mem_pickle = "fig2_mem_v3umk6fs.pkl"
with open(mem_pickle, "rb") as f:
    mem_df = pickle.load(f)


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
    "ELK": "tab:blue",
    "Quasi-ELK": "tab:purple",
}
lins = {"Sequential": ":", "DEER": "-", "Quasi-DEER": "-", "ELK": "-", "Quasi-ELK": "-"}
mars = {"Sequential": ".", "DEER": "+", "Quasi-DEER": "x", "ELK": "+", "Quasi-ELK": "x"}
tgs = {
    "Sequential": "seq",
    "DEER": "deer",
    "Quasi-DEER": "quasi",
    "ELK": "ELK",
    "Quasi-ELK": "qELK",
}
labs = [
    "Sequential",
    "DEER",
    "Quasi-DEER",
]  # We dont run these here: ['ELK', 'Quasi-ELK']

ls = [30_000, 100_000, 300_000, 1_000_000]
hs = [8, 16, 32, 64]


def make_fig_2(time_df, mem_df):
    fig, axes = plt.subplots(2, 4, sharex=True, figsize=(6, 3))

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
    yticks = [1e-1, 1, 1e1]
    yticklabels = ["$10^{-1}$", "$10^0$", "$10^1$"]
    for ax in axes[0]:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    # set y-tick lablels for mem
    yticks_mem = [1, 10]
    yticklabels_mem = ["$10^{0}$", "$10^{1}$"]
    for ax in axes[1]:
        ax.axhline(32, linestyle="--", c="k")
        ax.set_yticks(yticks_mem)
        ax.set_yticklabels(yticklabels_mem)

    # remove y tick lables
    for ax in axes[0, 1:]:
        ax.set_yticklabels([])

    for ax in axes[1, 1:]:
        ax.set_yticklabels([])

    # Top axis.
    for i, ax in enumerate(axes[0]):
        ax.set_title(f"$D = {hs[i]}$")
        ax.set_xticks(
            ls,
            ["30K", "100K", "300K", "1M"],
            rotation=90,
            ha="center",
            va="top",
        )
        ax.set_xlim((min(ls) * 0.9, max(ls) * 1.1))
        ax.set_ylim(0.01, 40)

    # Bottom axis
    for i, ax in enumerate(axes[1]):
        ax.axhline(16, linestyle="--", c="k")
        ax.set_xticks(
            ls,
            ["30K", "100K", "300K", "1M"],
            # rotation=90,
            ha="center",
            va="top",
        )
        ax.set_xlim((min(ls) * 0.9, max(ls) * 1.1))
        ax.set_ylim(0.5, 22)

    # Adjust the position of the x suplabel.
    t = fig.supxlabel(f"Sequence Length (T)", fontsize=MEDIUM_SIZE)
    t.set_position(
        (
            t.get_position()[0],
            t.get_position()[1] + 0.08,
        )
    )

    # Set each y-label up.
    axes[0, 0].set_ylabel("Wallclock (s)")
    axes[1, 0].set_ylabel("Memory (GB)", labelpad=9)

    # Now fake the legend.
    ax = axes[0, 3]
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
        bbox_to_anchor=(0, 1.6),
        bbox_transform=ax.transAxes,
    )

    # Format and save
    plt.tight_layout()
    fig.subplots_adjust(top=0.82, bottom=0.22, right=0.98)  # Adjusted bottom and top
    plt.savefig(
        "./fig2.pdf",
        bbox_inches="tight",
        pad_inches=0.01,
    )


if __name__ == "__main__":
    make_fig_2(time_df, mem_df)
