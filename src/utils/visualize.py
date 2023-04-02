import numpy as np
import torch
from matplotlib import colors as m_colors
from matplotlib import pyplot as plt
from torch import Tensor

from utils.data import normalize


def plot_series_samples(data: Tensor, n: int, labels: Tensor, seed: int = 42):
    """
    Plots a random set of 1-d time series.

    Parameters
    ----------
    :param Tensor data: A tensor containing the time series with shape(N, T)
    :param int n: The number of random samples to include in the plot
    :param Tensor labels: A tensor containing the label of each entry in data.
    :param int seed: random seed

    Returns
    --------

    """
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    x = np.arange(data.shape[1])
    torch.manual_seed(seed)
    idxs = torch.randint(data.shape[0], (n,))
    for idx in idxs:
        ax.plot(x, data[idx], label=f"{idx} ({'PD' if labels[idx] == 1.0 else 'HC'})")

    ax.set_ylim(-1, 1)
    ax.set_title('Focused Gaze - Samples of Processed Segments')
    ax.set_xlabel('Step')
    ax.set_ylabel('Normalized Gaze Position')
    ax.legend(ncol=2)
    plt.show()
    # fig.savefig('samples.png')


def visualize_latent_space(manifold, labels, class_names, show=True):
    colors = ['blue', 'darkorange', 'green']
    cmap = m_colors.ListedColormap(colors)

    fig, ax = plt.subplots()
    scatter = ax.scatter(manifold[:, 0], manifold[:, 1], c=labels, s=20, alpha=0.8, cmap=cmap)
    ax.legend(handles=scatter.legend_elements()[0], labels=class_names)
    if show:
        plt.show()
    return fig, ax


def plot_top_eigenvalues(test_features, n=100):
    normalized = normalize(test_features.detach())

    u, sigma, v_t = np.linalg.svd(normalized, full_matrices=False)

    # Compute eigenvalues from singular values
    eigenvalues = sigma ** 2 / np.sum(sigma ** 2)

    top_eigenvalues = eigenvalues[:n]
    explained_information = [sum(eigenvalues[:i]) for i in range(1, len(top_eigenvalues) + 1)]

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(np.arange(len(top_eigenvalues)), top_eigenvalues, label='explained variance')
    ax2.plot(np.arange(len(explained_information)), explained_information, label='cumulative explained variance', c='r')
    ax.set_title(f'Top {n} eigenvalues for ROCKET Embeddings')
    ax.legend(loc='lower right')
    ax2.legend(loc='center right')
    plt.show()
