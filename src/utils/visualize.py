import numpy as np
import pandas as pd
import torch
from matplotlib import colors as m_colors
from matplotlib import pyplot as plt
from matplotlib import colormaps as m_colormaps
from numpy import median
from pandas import DataFrame
from scipy.stats import median_absolute_deviation
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor, logical_and

from utils.data import normalize
from utils.ki import SACCADE, LABELS
from utils.path import figure_path

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


def visualize_latent_space(manifold, labels, class_names, show=True, title='Latent Neighborhood', cmap=None):
    colors = ['blue', 'darkorange']
    if cmap is None:
        cmap = m_colors.ListedColormap(colors)

        fig, ax = plt.subplots(figsize=(3.8, 3))
        scatter = ax.scatter(manifold[:, 0], manifold[:, 1], c=labels, s=20, alpha=0.8, cmap=cmap)
    
        ax.legend(handles=scatter.legend_elements()[0], labels=class_names, loc='upper right')
    else:
        fig, ax = plt.subplots(figsize=(3.8, 3))
        scatter = ax.scatter(manifold[:, 0], manifold[:, 1], c=labels, s=20, alpha=0.8, cmap=cmap, vmin=-1.5, vmax=1.5) 
        cbaxes = inset_axes(ax, width="30%", height="5%", loc=1)
        fig.colorbar(scatter, cax=cbaxes, ticks=[-1,1], orientation='horizontal', label='Model Score')

    # Remove the ticks from
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if show:
        plt.show()
    return fig, ax, xlim, ylim

def visualize_latent_space_medication(manifold, labels, attribute, class_names, show=True, title='Latent Neighborhood', cmap=None):

    hc = logical_and(labels == 0, attribute == LABELS['HC'])
    pd_off = logical_and(labels == 1, attribute == LABELS['PDOFF'])
    pd_on = logical_and(labels == 1, attribute == LABELS['PDON'])

    if cmap is not None:

        # Get colors from cmap
        color_map = m_colormaps[cmap]
        colors = [color_map(0.0), color_map(0.5), color_map(1.0)]

        # Fixed colors (change darkorange for other two colors)
        #colors = ['blue', 'limegreen', 'indianred']

        fig, ax = plt.subplots(figsize=(3.8, 3))
        scatter_hc = ax.scatter(manifold[hc, 0], manifold[hc, 1], s=20, cmap=cmap, alpha=0.8, color=colors[0], marker='o')

        # Plot with full circles for PD Off
        scatter_pd_off = ax.scatter(manifold[pd_on, 0], manifold[pd_on, 1], s=20, alpha=0.8, color=colors[1], marker='o')

        # Plot with empty circles for PD On
        scatter_pd_on = ax.scatter(manifold[pd_off, 0], manifold[pd_off, 1], s=20, alpha=0.8, color=colors[2], marker='o')

    else:
        colors = ['blue', 'darkorange']
        cmap = m_colors.ListedColormap(colors)

        fig, ax = plt.subplots(figsize=(3.8, 3))

        scatter_hc = ax.scatter(manifold[hc, 0], manifold[hc, 1], s=20, cmap=cmap, alpha=0.8,color='blue', marker='o')

        # Plot with full circles for PD Off
        scatter_pd_off = ax.scatter(manifold[pd_off, 0], manifold[pd_off, 1], s=20, alpha=0.8,color='darkorange', marker='o')

        # Plot with empty circles for PD On 
        scatter_pd_on = ax.scatter(manifold[pd_on, 0], manifold[pd_on, 1], s=20, alpha=0.8,cmap=cmap, facecolors='none', edgecolors='darkorange')

    ax.legend((scatter_hc, scatter_pd_on, scatter_pd_off), ('HC', 'PD On', 'PD Off'), loc='upper right')

    # Remove the ticks from
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if show:
        plt.show()
    return fig, ax, xlim, ylim

def separate_latent_space_by_attr(manifold, labels, attribute, class_names, attribute_names, show=True, title=''):
    colors = ['blue', 'darkorange']
    cmap = m_colors.ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(3.8, 3))

    hc_pro = logical_and(labels == 0, attribute == SACCADE['pro'])
    hc_anti = logical_and(labels == 0, attribute == SACCADE['anti'])
    pd_pro = logical_and(labels == 1, attribute == SACCADE['pro'])
    pd_anti = logical_and(labels == 1, attribute == SACCADE['anti'])

    # Plot the latent space with the labels as colors and the attribute as the marker
    scatter_hc_pro = ax.scatter(manifold[hc_pro, 0], manifold[hc_pro, 1], s=20, cmap=cmap, alpha=0.8,color='blue', marker='^')

    scatter_hc_anti = ax.scatter(manifold[hc_anti, 0], manifold[hc_anti, 1], s=20,      alpha=0.8,cmap=cmap, marker='x', color='blue')

    scatter_pd_pro = ax.scatter(manifold[pd_pro, 0], manifold[pd_pro, 1], s=20, alpha=0.8,
    color='darkorange', marker='^')
    
    scatter_pd_anti = ax.scatter(manifold[pd_anti, 0], manifold[pd_anti, 1], c='darkorange', s=20, alpha=0.8,cmap=cmap, marker='x')

    ax.legend((scatter_hc_pro, scatter_hc_anti, scatter_pd_pro, scatter_pd_anti),
              ('HC Pro', 'HC Anti', 'PD Pro', 'PD Anti'), loc='upper left', ncol=2)

    ax.set_title(title)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    # Save figure as svg
    fig.savefig('latent_representation_saccade.svg', format='svg', dpi=1200)

    if show:
        plt.show()
    return fig, ax


# Uses a metadata dataframe to set the opacity based on a series (key)
def t_sne_subject_metadata(manifold, labels, subject_ids, class_names, metadata,
                           show=True, title='Latent Neighborhood',
                           key='age',xlim=None, ylim=None):
    colors = ['blue', 'darkorange']
    cmap = m_colors.ListedColormap(colors)

    # Create mask for all subjects in the data that has metadata.
    has_metadata = np.isin(subject_ids, metadata['ID'])
    # Mask the rows
    manifold = manifold[has_metadata]
    labels = labels[has_metadata]
    subject_ids = subject_ids[has_metadata]

    # Reorder and replicate rows in df based on subject IDs for the datapoints above
    reordered_df = pd.concat([metadata[metadata.ID == s_id] for s_id in subject_ids.tolist()])

    # Create array to set alpha for every trial based on the age series in metadata for the corresponding ID.
    # Should create values in range [0.5 - 1]
    #min_subtracted_values = reordered_df[key] - reordered_df[key].min()
    #normalized_values = 0.5 + (min_subtracted_values / min_subtracted_values.max()) / 2
    normalized_values = reordered_df[key]

    fig, ax = plt.subplots(figsize=(3.8, 3))
    #scatter = ax.scatter(manifold[:, 0][labels == LABELS['HC']], manifold[:, 1][labels == LABELS['HC']],
    #                     c=normalized_values.values[labels == LABELS['HC']], s=20, marker='x')  # ,
    scatter = ax.scatter(manifold[:, 0][labels == LABELS['PDOFF']], manifold[:, 1][labels == LABELS['PDOFF']],
                         c=normalized_values.values[labels == LABELS['PDOFF']], s=20, marker='o', alpha=0.8 )  # ,
    # cmap=cmap)
    # ax.legend(handles=scatter.legend_elements()[0], labels=class_names, loc='upper left')
    # Remove the ticks from
    # Min and max of the normalized values
    min_val = normalized_values.min()
    max_val = normalized_values.max()

    cbaxes = inset_axes(ax, width="30%", height="5%" , loc=1)
    fig.colorbar(scatter, cax=cbaxes, orientation='horizontal', label=key, ticks=[min_val,max_val])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    print(xlim, ylim)

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
    ax.set_ylim(-0.1, 0.1)
    ax.set_title('Trial Data')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gaze Position (a.u.)')
    ax.legend(ncol=2, loc='lower left')
    plt.tight_layout()
    fig.savefig(figure_path.joinpath('trial_data.svg'), format='svg', dpi=1200)
    plt.show()


def plot_confusion_matrix(labels, predictions, class_names, title='Confusion Matrix', filename='', show=True):
    """
    Plot a confusion matrix from a list of labels and predictions

    :param labels: The true labels
    :param predictions: The predicted labels
    :param class_names: The names of the classes
    :param title: The title of the plot
    :param filename: The filename to save the plot to. The plot will be saved as an svg.
    :param show: Whether to show the plot
    :return: The figure
    """
    figure = ConfusionMatrixDisplay.from_predictions(labels, predictions, display_labels=class_names, cmap='Blues')
    figure.ax_.set_title(title)
    figure.figure_.set_size_inches(4, 3)
    plt.tight_layout()
    if len(filename):
        figure.figure_.savefig(f'{figure_path.joinpath(filename)}.svg', format='svg', dpi=1200)
    if show:
        plt.show()

    return figure


def plot_latent_neighborhood(features, batch, class_names, filename='', show=False, metadata=None,
                             key='age', scores=None):
    """
    Plot a 2d TSNE embedding of the features computed from the batch
    :param features: The high dimensional features from the model
    :param batch: The batch of data from which the features were computed
    :param class_names: The names of the classes in the dataset
    :param filename: The filename to save the plot to. If a filename is provided, the plot will be saved as an svg.
    :param show: Whether to show the plot.
    :param metadata: Subject metadata as an optional dataframe. If passed, will visualize based on key
    :param key: The metadata key to visualize with the t-sne plot
    :param scores: The scores in range [0,1] for each datapoint.
    """
    tsne = TSNE(n_components=2, perplexity=30)
    manifold = tsne.fit_transform(features.detach())
    fig, _ = separate_latent_space_by_attr(manifold, batch.y, batch.s, class_names, SACCADE.keys(), title='Latent Representation of Test Set')

    fig, ax, xlim, ylim = visualize_latent_space(manifold, batch.y, class_names, show=False, title='Subject Label')

    plt.tight_layout()
    if len(filename):
        fig.savefig(f'{figure_path.joinpath(filename)}_classes.pdf', format='pdf', dpi=1200)
    if show:
        plt.show()

    fig, ax, xlim, ylim = visualize_latent_space_medication(manifold, batch.y, batch.g, class_names, show=False, title='Subject Label', cmap='viridis')

    plt.tight_layout()
    if len(filename):
        fig.savefig(f'{figure_path.joinpath(filename)}_medication.pdf', format='pdf', dpi=1200)
    if show:
        plt.show()

    if scores is not None:
        fig, ax, xlim, ylim = visualize_latent_space(manifold, scores, class_names, show=False,
                                         title="Model's Confidence Score (a.u.)", cmap='viridis')
        plt.tight_layout()
        if len(filename):
            fig.savefig(f'{figure_path.joinpath(filename)}_model_scores.pdf', format='pdf', dpi=1200)
        if show:
            plt.show()

    if metadata is not None:
        print(xlim, ylim)
        fig, ax = t_sne_subject_metadata(manifold, batch.y, batch.z, class_names, metadata, show=True,
                                            title = 'Disease Duration (years)', # 'Disease Duration (years)'
                                            key=key, xlim=xlim, ylim=ylim)

        if len(filename):
            fig.savefig(f'{figure_path.joinpath(filename)}_metadata.pdf', format='pdf', dpi=1200)
        if show:
            plt.show()


def grouped_bar_chart():
    import seaborn as sns
    import matplotlib.pyplot as plt

    ROCKET = 'Rocket'
    IT = 'InceptionTime'
    DROCKET = 'Detach-Rocket'
    Brien = 'Brien et al.'
    TRIAL = 'Trial'
    SUBJECT = 'Subject'

    df = DataFrame(columns=['Model', 'Level', 'Accuracy'],
                   data=[
                       [IT, TRIAL, 51.82],
                       [IT, TRIAL, 59.32],
                       [IT, TRIAL, 55.00],
                       [IT, TRIAL, 55.00],
                       [IT, TRIAL, 57.50],
                       [IT, SUBJECT, 68.75],
                       [IT, SUBJECT, 81.25],
                       [IT, SUBJECT, 81.25],
                       [IT, SUBJECT, 75.00],
                       [IT, SUBJECT, 81.25],
                       [ROCKET, TRIAL, 66.14],
                       [ROCKET, TRIAL, 65.91],
                       [ROCKET, TRIAL, 67.95],
                       [ROCKET, TRIAL, 68.86],
                       [ROCKET, TRIAL, 71.36],
                       [ROCKET, SUBJECT, 87.5],
                       [ROCKET, SUBJECT, 87.5],
                       [ROCKET, SUBJECT, 81.25],
                       [ROCKET, SUBJECT, 87.5],
                       [ROCKET, SUBJECT, 93.75],
                       [DROCKET, TRIAL, 73.86],
                       [DROCKET, TRIAL, 73.41],
                       [DROCKET, TRIAL, 74.32],
                       [DROCKET, TRIAL, 72.05],
                       [DROCKET, TRIAL, 73.64],
                       [DROCKET, SUBJECT, 100.00],
                       [DROCKET, SUBJECT, 93.75],
                       [DROCKET, SUBJECT, 93.75],
                       [DROCKET, SUBJECT, 93.75],
                       [DROCKET, SUBJECT, 100.00],
                       [DROCKET, SUBJECT, 100.00],
                       [Brien, SUBJECT, 86.8],
                       [Brien, SUBJECT, 77.3],
                   ])

    # Creating the bar chart
    plt.figure(figsize=(5, 3))
    plt.grid(axis='y')
    sns.barplot(data=df, x='Level', y='Accuracy', hue='Model', capsize=0.05, errorbar='sd')
    plt.title('Accuracy of Models by Classification Level')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Classification Level')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2)
    plt.tight_layout()
    plt.ylim(0, 105)

    plt.show()
