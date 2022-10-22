import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

plt.rcParams.update({"font.size": 16})


def plot_means_stds(ax: Axes, means_stds: np.ndarray, batch_sizes: list, ylims: list = None):
    """Plots means ± stds vs batch size

    Args:
        ax (Axes): matplotlib axis on which to plot
        means_stds (np.ndarray): ``[N, 2]`` - first column is means, second column is stds
        batch_sizes (list): list of length N
        ylims (list): plot y limits
    """
    means = np.nan_to_num(means_stds[:, 0])[: len(batch_sizes)]
    stds = np.nan_to_num(means_stds[:, 1])[: len(batch_sizes)]

    ax.plot(
        batch_sizes,
        means,
        marker="o",
        linestyle="--",
    )

    ax.fill_between(batch_sizes, means - stds, means + stds, alpha=0.2)
    ax.set_ylim(ylims)
    ax.set_xlabel("N")


def metric_label(ax: Axes, label: str):
    """Adds vertical ``label`` annotation to ``ax``"""
    ax.annotate(
        label,
        xy=(0, -1),
        xytext=(-ax.yaxis.labelpad - 15, 0),
        xycoords=ax.yaxis.label,
        textcoords="offset points",
        ha="right",
        va="center",
        rotation=90,
    )
