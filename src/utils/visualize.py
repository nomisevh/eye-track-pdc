import numpy as np
import torch
from matplotlib import colors as m_colors
from matplotlib import pyplot as plt
from numpy import median
from pandas import DataFrame
from scipy.stats import median_absolute_deviation
from torch import Tensor, logical_and

from utils.data import normalize
from utils.ki import SACCADE
from utils.path import figure_path


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


def visualize_latent_space(manifold, labels, class_names, show=True, title='Latent Neighborhood'):
    colors = ['blue', 'darkorange']
    cmap = m_colors.ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(3.8, 3))
    scatter = ax.scatter(manifold[:, 0], manifold[:, 1], c=labels, s=20, alpha=0.8, cmap=cmap)
    ax.legend(handles=scatter.legend_elements()[0], labels=class_names, loc='upper left')
    # Remove the ticks from
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    if show:
        plt.show()
    return fig, ax


def separate_latent_space_by_attr(manifold, labels, attribute, class_names, attribute_names, show=True, title=''):
    colors = ['blue', 'darkorange']
    cmap = m_colors.ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(3.8, 3))

    hc_pro = logical_and(labels == 0, attribute == SACCADE['pro'])
    hc_anti = logical_and(labels == 0, attribute == SACCADE['anti'])
    pd_pro = logical_and(labels == 1, attribute == SACCADE['pro'])
    pd_anti = logical_and(labels == 1, attribute == SACCADE['anti'])

    # Plot the latent space with the labels as colors and the attribute as the marker
    scatter_hc_pro = ax.scatter(manifold[hc_pro, 0], manifold[hc_pro, 1], s=20, cmap=cmap, alpha=0.8,
                                color='blue', marker='^')
    scatter_hc_anti = ax.scatter(manifold[hc_anti, 0], manifold[hc_anti, 1], s=20, alpha=0.8,
                                 cmap=cmap, marker='x', color='blue')
    scatter_pd_pro = ax.scatter(manifold[pd_pro, 0], manifold[pd_pro, 1], s=20, alpha=0.8,
                                color='darkorange', marker='^')
    scatter_pd_anti = ax.scatter(manifold[pd_anti, 0], manifold[pd_anti, 1], c='darkorange', s=20, alpha=0.8,
                                 cmap=cmap, marker='x')

    ax.legend((scatter_hc_pro, scatter_hc_anti, scatter_pd_pro, scatter_pd_anti),
              ('HC Pro', 'HC Anti', 'PD Pro', 'PD Anti'), loc='upper left', ncol=2)

    ax.set_title(title)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    plt.tight_layout()
    # Save figure as svg
    fig.savefig('latent_representation_saccade.svg', format='svg', dpi=1200)

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


def histogram_heuristic(h_values, h_name, threshold=3.0, channels=''):
    med = median(h_values)
    mad = median_absolute_deviation(h_values)
    # Plot a histogram of the heuristic values with vertical lines for the median and threshold * MAD
    plt.hist(h_values, bins=100)
    plt.axvline(med, color='r')
    plt.axvline(med + threshold * mad, color='g')
    plt.axvline(med - threshold * mad, color='g')
    plt.title(f'{channels} {h_name} histogram')
    plt.show()


def plot_inter_saccade(dataframe: DataFrame):
    target_diff = dataframe['target'].diff().fillna(0)
    # The start and end of each saccade is marked by a change in the target position
    anchors = [*dataframe['target'][target_diff != 0].index.tolist()]
    to_plot = dataframe.iloc[anchors[0]:anchors[3]]
    # to_plot = dataframe

    to_plot['Time (s)'] = to_plot['Time (ms)'] / 1000
    dataframe['Time (s)'] = dataframe['Time (ms)'] / 1000

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(to_plot['Time (s)'], to_plot['position'], label='gaze x')
    ax.plot(to_plot['Time (s)'], to_plot['drift'], label='gaze y')
    ax.plot(to_plot['Time (s)'], to_plot['target'], label='target x')
    # Highlight from anchor 2 - 440 data points to anchor 2
    ax.axvspan(dataframe['Time (s)'][anchors[2] - 440], dataframe['Time (s)'][anchors[2]], alpha=0.2, color='black')

    # Highlight from 1000 ms after anchor 1 to anchor 2
    # ax.axvspan(dataframe['Time (ms)'][anchors[1]] + 1000, dataframe['Time (ms)'][anchors[2]], alpha=0.2, color='black')
    # Plot a horizontal line at three standard deviations above the mean
    # ax.axhline(dataframe['position'].mean() + 3 * dataframe['position'].std(), color='r', linestyle='--')
    # ax.set_title('Antisaccade Session Data')
    ax.set_title('Fixation Period')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Gaze Position (a.u.)')
    ax.legend(loc='lower left')
    plt.tight_layout()
    fig.savefig('fixation_period.svg', format='svg', dpi=1200)
    plt.show()

    # a.u. = arbitrary units


def plot_trial(mts):
    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.arange(mts.x.shape[1])
    y = mts.x
    ax.plot(x, y[0], label='gaze x', linewidth=1)
    ax.plot(x, y[1], label='gaze y', linewidth=1)
    ax.plot(x, y[2], label='gaze x vel', linewidth=1, c='brown')
    ax.plot(x, y[3], label='gaze y vel', linewidth=1, c='purple')
    ax.set_ylim(-0.3, 0.3)
    ax.set_title('Trial Data')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gaze Position (a.u.)')
    ax.legend(ncol=2, loc='lower left')
    plt.tight_layout()
    fig.savefig(figure_path.joinpath('trial_data.svg'), format='svg', dpi=1200)
    plt.show()
