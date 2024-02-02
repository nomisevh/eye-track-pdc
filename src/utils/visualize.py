import numpy as np
import torch
from matplotlib import colors as m_colors
from matplotlib import pyplot as plt
from numpy import median
from pandas import DataFrame
from scipy.stats import median_absolute_deviation
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
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


def plot_latent_neighborhood(features, batch, class_names, filename='', show=False):
    """
    Plot a 2d TSNE embedding of the features computed from the batch
    :param features: The high dimensional features from the model
    :param batch: The batch of data from which the features were computed
    :param class_names: The names of the classes in the dataset
    :param filename: The filename to save the plot to. If a filename is provided, the plot will be saved as an svg.
    :param show: Whether to show the plot.
    """
    tsne = TSNE(n_components=2, perplexity=30)
    manifold = tsne.fit_transform(features.detach())
    fig, _ = separate_latent_space_by_attr(manifold, batch.y, batch.s, class_names, SACCADE.keys(),
                                           title='Latent Representation of Test Set')

    fig, ax = visualize_latent_space(manifold, batch.y, class_names, show=False,
                                     title='Latent Representation of Test Set')
    plt.tight_layout()
    if len(filename):
        fig.savefig(f'{figure_path.joinpath(filename)}.svg', format='svg', dpi=1200)
    if show:
        plt.show()


def plot_results_bar_chart():
    import matplotlib.pyplot as plt
    import numpy as np

    # Data from the user's table
    models = ["ROCKET", "Detach-ROCKET", "InceptionTime", "Brien et al."]
    levels = ['Trial', 'Subject']
    trial_std_dev = [2.23, 0.85, 2.84]
    subject_accuracies = [87.50, 96.25, 77.50, 82.00]
    subject_std_dev = [4.42, 3.42, 5.59, 6.7]
    rocket_accuracies = []

    x = np.arange(2)  # the label locations
    x_trial = np.arange(len(trial_std_dev))
    x_sub = np.arange(len(subject_std_dev))
    width = 0.35  # the width of the bars

    # Adjusting the plot to include only Subject Accuracy for "Brien et al."
    # and improving the aesthetics of the plot as requested

    # Update data to remove Trial Accuracy for "Brien et al."
    trial_accuracies = [68.04, 73.46, 55.73]  # None for Brien et al.

    # Define colors
    trial_color = 'blue'
    subject_color = 'orange'

    # Create a new figure with slightly larger size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars for Trial Accuracy except for "Brien et al."
    trial_bars = ax.bar(x_trial - width / 2, trial_accuracies, width, yerr=trial_std_dev,
                        label='Trial Accuracy', color=trial_color, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})

    # Create bars for Subject Accuracy
    subject_bars = ax.bar(x_sub + width / 2, subject_accuracies, width, yerr=subject_std_dev,
                          label='Subject Accuracy', color=subject_color, capsize=5,
                          error_kw={'elinewidth': 2, 'capthick': 2})

    # Improve the aesthetics
    ax.set_xlabel('Classification Level', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Model Accuracy on Trial and Subject Level', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=14)
    ax.legend(fontsize=14)

    # Rotate the tick labels for better readability
    # plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='center')

    # Set the style of the grid
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    # Set the background color
    ax.set_facecolor('white')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create bars with rounded corners for Trial Accuracy

    rocket_bars = ax.bar(x_trial - width / 2, trial_accuracies, width, yerr=trial_std_dev, label='Trial Accuracy',
                         color=trial_color,

                         capsize=5, error_kw=dict(elinewidth=2, ecolor='black'),

                         align='center', zorder=2)

    # Create bars with rounded corners for Subject Accuracy

    inception_bars = ax.bar(x_sub + width / 2, subject_accuracies, width, yerr=subject_std_dev,
                            label='Subject Accuracy',
                            color=subject_color,

                            capsize=5, error_kw=dict(elinewidth=2, ecolor='black'),

                            align='center', zorder=2)
    detach_bars = ax.bar(x_sub + width / 2, subject_accuracies, width, yerr=subject_std_dev, label='Subject Accuracy',
                         color=subject_color,

                         capsize=5, error_kw=dict(elinewidth=2, ecolor='black'),

                         align='center', zorder=2)
    brien_bars = ax.bar(x_sub + width / 2, subject_accuracies, width, yerr=subject_std_dev, label='Subject Accuracy',
                        color=subject_color,

                        capsize=5, error_kw=dict(elinewidth=2, ecolor='black'),

                        align='center', zorder=2)

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, -40),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)

    # Call the function to add labels

    add_labels(trial_bars)

    add_labels(subject_bars)

    # Set the corner style to round

    for bar in trial_bars:
        bar.set_capstyle('round')

    for bar in subject_bars:
        bar.set_capstyle('round')

    # Show the figure with the new style
    plt.tight_layout()
    plt.show()


def plot_grouped_bar_chart():
    import matplotlib.pyplot as plt
    import numpy as np

    # Data
    models = ["ROCKET", "Detach-ROCKET", "InceptionTime", "Brien et al."]
    trial_accuracies = [68.04, 73.46, 55.73, np.nan]  # NaN for Brien et al.
    subject_accuracies = [87.50, 96.25, 77.50, 82.00]
    trial_std_dev = [2.23, 0.85, 2.84, np.nan]  # NaN for Brien et al.
    subject_std_dev = [4.42, 3.42, 5.59, 6.7]

    # X-axis locations for the groups
    ind = np.arange(len(models))
    width = 0.35

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting bars for Trial and Subject accuracies
    trial_bars = ax.bar(ind - width / 2, trial_accuracies, width, yerr=trial_std_dev,
                        label='Trial Accuracy', color='blue', capsize=5)
    subject_bars = ax.bar(ind + width / 2, subject_accuracies, width, yerr=subject_std_dev,
                          label='Subject Accuracy', color='orange', capsize=5)

    # Aesthetics
    ax.set_xlabel('Accuracy Type', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Model Accuracy for Trial and Subject', fontsize=16)
    ax.set_xticks(ind)
    ax.set_xticklabels(models, fontsize=14)
    ax.legend(fontsize=14)

    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def grouped_bar_chart():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Data provided
    models = ["ROCKET", "Detach-ROCKET", "InceptionTime", "Brien et al."]
    trial_accuracies = [68.04, 73.46, 55.73, None]  # None for Brien et al.
    trial_std_dev = [2.23, 0.85, 2.84, None]
    subject_accuracies = [87.50, 96.25, 77.50, 82.00]
    subject_std_dev = [4.42, 3.42, 5.59, 6.7]

    # Creating a DataFrame
    data = []
    for model, t_acc, t_std, s_acc, s_std in zip(models, trial_accuracies, trial_std_dev, subject_accuracies,
                                                 subject_std_dev):
        data.append({'Model': model, 'Level': 'Trial', 'Accuracy': t_acc, 'StdDev': t_std})
        data.append({'Model': model, 'Level': 'Subject', 'Accuracy': s_acc, 'StdDev': s_std})

    df = pd.DataFrame(data)

    # Function to map values to min-max interval for error bars
    def map_to_error(series):
        mean = series
        return mean - 5, mean + 5

    # Creating the bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Level', y='Accuracy', hue='Model', capsize=0.1)
    plt.title('Accuracy of Models by Classification Level')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Classification Level')

    # Bottom error, then top
    yerr = [df['Accuracy'] - df['StdDev'], df['Accuracy'] + df['StdDev']]

    plt.errorbar(x=[0, 1, 2, 3, 4, 5, 6, 7], y=df['Accuracy'],
                 yerr=yerr, fmt='none', c='r')

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

    # Data provided
    models = ["ROCKET", "Detach-ROCKET", "InceptionTime", "Brien et al."]
    trial_accuracies = [68.04, 73.46, 55.73, None]  # None for Brien et al.
    trial_std_dev = [2.23, 0.85, 2.84, None]
    subject_accuracies = [87.50, 96.25, 77.50, 82.00]
    subject_std_dev = [4.42, 3.42, 5.59, 6.7]

    # Creating a DataFrame
    # data = []
    # for model, t_acc, t_std, s_acc, s_std in zip(models, trial_accuracies, trial_std_dev, subject_accuracies,
    #                                             subject_std_dev):
    #    data.append({'Model': model, 'Level': 'Trial', 'Accuracy': t_acc, 'StdDev': t_std})
    #    data.append({'Model': model, 'Level': 'Subject', 'Accuracy': s_acc, 'StdDev': s_std})
    #
    # df = pd.DataFrame(data)

    # Function to map values to min-max interval for error bars
    def map_to_error(series):
        mean = series
        return mean - 5, mean + 5

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

    # Bottom error, then top
    # yerr = [df['Accuracy'] - df['StdDev'], df['Accuracy'] + df['StdDev']]
    #
    # plt.errorbar(x=[0, 1, 2, 3, 4, 5, 6, 7], y=df['Accuracy'],
    #             yerr=yerr, fmt='none', c='r')

    plt.show()
