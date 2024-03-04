"""
Interprets the kernels and classifier weights of the ROCKET model and its ridge classifier.

This file loads the best ROCKET model and its classifier, and computes feature importance based on Sequential Feature
Detachment. The optimal selection of kernels is then analyzed based on the kernel dilation and length.
"""
import pickle
import types

import matplotlib.colors as m_colors
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from torch import nn, tensor
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from models.rocket import dissected_forward, ROCKET
from utils.interpretability import feature_detachment, select_optimal_model, retrain_optimal_model
from utils.misc import set_random_state
from utils.path import config_path, rocket_instances_path, figure_path


def main(seed, correct_way=True, use_cached_features=True):
    set_random_state(seed)

    # Load configs
    with open(config_path.joinpath('leif.yaml')) as reader:
        processor_config = load_yaml(reader, Loader=FullLoader)
    with open(config_path.joinpath('rocket.yaml')) as reader:
        rocket_config = load_yaml(reader, Loader=FullLoader)

    dm = KIDataModule(processor_config=processor_config,
                      bundle_as_experiments=False,
                      exclude=['vert'],
                      binary_classification=True,
                      batch_size=-1)
    dm.setup('fit')
    dm.setup('test')

    # Create the rocket model
    rocket = ROCKET(c_in=dm.train_ds.x.shape[1],
                    seq_len=dm.train_ds.x.shape[2],
                    **rocket_config)

    ridge_clf = RidgeClassifier(alpha=1e3)

    # Batch is entire dataset
    train_batch = next(iter(dm.train_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))

    def features_filename(partition):
        return rocket_instances_path.joinpath(f'rocket_{seed}_{partition}_features.ckpt')

    try:
        if not use_cached_features:
            raise FileNotFoundError
        print(f'reusing features from {features_filename("partition")}')
        train_features = torch.load(features_filename('train'))
        test_features = torch.load(features_filename('test'))
        val_features = torch.load(features_filename('val'))
        rocket.train = False
    except FileNotFoundError:
        # Perform ROCKET transformation stage on train, val, and test data
        rocket.train = True
        train_features = rocket(train_batch.x).numpy()
        rocket.train = False
        test_features = rocket(test_batch.x).numpy()
        val_features = rocket(val_batch.x).numpy()
        torch.save(train_features, features_filename('train'))
        torch.save(test_features, features_filename('test'))
        torch.save(val_features, features_filename('val'))

    ridge_clf.fit(train_features, train_batch.y)

    # Make predictions on validation set
    train_pred = ridge_clf.predict(train_features)
    test_pred = ridge_clf.predict(test_features)
    val_pred = ridge_clf.predict(val_features)

    # The RidgeClassifier maps the targets to {-1, 1}, but our labels are {0, 1}
    train_pred[train_pred < 0] = 0
    test_pred[test_pred < 0] = 0
    val_pred[val_pred < 0] = 0

    # Targets may be accessed like this
    train_labels = train_batch.y
    test_labels = test_batch.y
    val_labels = val_batch.y

    # concatenate train and val features and labels
    train_val_features = np.concatenate((train_features, val_features), axis=0)
    train_val_labels = np.concatenate((train_labels, val_labels), axis=0)

    # Compute balanced accuracy
    train_score = balanced_accuracy_score(train_labels, train_pred)
    test_score = balanced_accuracy_score(test_labels, test_pred)
    val_score = balanced_accuracy_score(val_labels, val_pred)

    if correct_way:
        # TODO this comment seems backward
        # CORRECT WAY (train and val are combined)
        percentage_vector, score_list_train, score_list_val, feature_importance_matrix, feature_selection_matrix = \
            feature_detachment(
                ridge_clf,
                train_features,
                val_features,
                train_labels,
                val_labels,
                total_number_steps=175)

        optimal_index, optimal_percentage = select_optimal_model(percentage_vector, score_list_val, score_list_val[0])

    else:
        # INCORRECT WAY (train and val are not combined)
        percentage_vector, score_list_train, score_list_test, feature_importance_matrix, feature_selection_matrix = \
            feature_detachment(
                ridge_clf,
                train_val_features,
                test_features,
                train_val_labels,
                test_labels)

        optimal_index, optimal_percentage = select_optimal_model(percentage_vector, score_list_test, score_list_test[0])

    # call retrain optimal model
    retrained_clf = retrain_optimal_model(feature_selection_matrix, train_val_features, test_features,
                                          train_val_labels, test_labels, train_score, test_score, ridge_clf.alpha,
                                          optimal_index, test_batch)

    # Print the optimal percentage of features in percentage style with 2 decimal places
    print(' ')
    print('Used percentage of features: ' + str(round(100 * optimal_percentage, 2)) + '%')

    index_is_even = np.arange(len(feature_selection_matrix[optimal_index])) % 2 == 0
    kept_is_max = index_is_even[feature_selection_matrix[optimal_index]]

    # Reshape by kernel (row-first), there are 2 features per kernel
    feature_selection = feature_selection_matrix[optimal_index].reshape((-1, 2))  # (num_kernels, 2)
    # Keep all kernels where at least one feature was selected as optimal
    keep_mask = feature_selection.any(axis=1)
    kept_kernels = nn.ModuleList([kernel for kernel, keep in zip(rocket.convs, keep_mask) if keep])

    # Find the first correctly predicted data point that is PD
    sample_pd_index = torch.logical_and(test_batch.y == tensor(test_pred), test_batch.y == 1).nonzero().squeeze()[0]
    # Find the first correctly predicted data point that is HC
    sample_hc_index = torch.logical_and(test_batch.y == tensor(test_pred), test_batch.y == 0).nonzero().squeeze()[0]

    compare_kernel_distributions(rocket.convs, kept_kernels)

    # Limit the kernels in the rocket model
    rocket = apply_pruning(rocket, kept_kernels, feature_selection_matrix[optimal_index], feature_selection[keep_mask])

    torch.save(rocket, rocket_instances_path.joinpath(f'pruned_rocket_{seed}.ckpt'))
    with open(rocket_instances_path.joinpath(f'pruned_rocket_clf_{seed}.pkl'), 'wb') as writer:
        pickle.dump(retrained_clf[3], writer)

    # Use None indexing to keep dimensionality of x
    # dissect_rocket_transformation_stage(test_batch.x[sample_pd_index:sample_pd_index + 1], rocket)


# This is a mess, will fix later
def apply_pruning(rocket, kept_kernels, kept_features, features_per_kernel):
    rocket.convs = kept_kernels
    rocket.n_kernels = len(kept_kernels)
    rocket.mean = rocket.mean[:, kept_features]
    rocket.std = rocket.std[:, kept_features]
    for features, kernel in zip(features_per_kernel, rocket.convs):
        kernel.use_features = features
    return rocket


def compare_kernel_distributions(all_kernels, kept_kernels):
    # Histogram the distribution of kernel dilation
    kept_dilation = [k.dilation[0] for k in kept_kernels]
    all_dilation = [k.dilation[0] for k in all_kernels]
    hist_kernel_property(kept_dilation, all_dilation, 'Dilation')

    # Do the same for kernel length
    kept_len = [k.weight.shape[-1] for k in kept_kernels]
    all_len = [k.weight.shape[-1] for k in all_kernels]
    hist_kernel_property(kept_len, all_len, 'Length')

    # And the bias
    # kept_bias = [k.bias.item() for k in kept_kernels]
    # all_bias = [k.bias.item() for k in all_kernels]
    # hist_kernel_property(kept_bias, all_bias, 'Bias')

    # Receptive field
    kept_rf = [k.dilation[0] * k.weight.shape[-1] for k in kept_kernels]
    all_rf = [k.dilation[0] * k.weight.shape[-1] for k in all_kernels]
    hist_kernel_property(kept_rf, all_rf, 'Receptive Field')


def hist_kernel_property(kept_kernels_prop, all_kernels_prop, prop_name):
    fig, ax = plt.subplots()
    # Todo there is some problem with Doane binning, which results in the histogram not summing to 1.
    ax.hist([kept_kernels_prop, all_kernels_prop], bins='doane', label=['Kept', 'All Kernels'], density=True)
    ax.legend(loc='upper right')
    ax.set_title(f"Kernel {prop_name}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    plt.show()


def dissect_rocket_transformation_stage(x, rocket):
    """
    Dissect the rocket transformation stage by visualizing the output from each kernel in the convolution op. alongside
    the input for a data sample x.
    :param x: A tensor holding the input signal of shape (1, M, T).
    :param rocket: The rocket model to be used.
    """
    # Overwrite the forward pass to also provide the output signal before computing PPV and Max
    rocket.forward = types.MethodType(dissected_forward, rocket)
    features, out_signal = rocket(x)  # Output is shape (1, k, T)  where K is the number of kernels
    num_kernels = out_signal.shape[1]

    # For every kernel, visualize the convolution and stack them horizontally in figure.
    fig, axes = plt.subplots(num_kernels, 1, figsize=(8, 2 * num_kernels))
    common_handles = []
    for i in range(num_kernels):
        handles = visualize_convolution(x, out_signal[:, i], axes[i], rocket.convs[i].padding)
        if i == 0:
            common_handles = handles

    # Add a common legend for all subplots.
    fig.legend(handles=common_handles)
    fig.suptitle('Rocket Transformation Visualization')
    fig.savefig(figure_path.joinpath('pd1_rocket_convolution_viz.svg'), dpi=350)
    plt.show()


def visualize_convolution(x, output_signal, ax, padding):
    """
    Visualize the various dims of an input series and the output from one kernel of the rocket convolution operation.
    :param x: A tensor holding the input signal of shape (1, M, T).
    :param output_signal: The output signal from the convolution of one kernel. (1, T)
    :param ax: The matplotlib axis to plot on.
    :param padding: The padding applied to each side of the input for this specific kernel.
    """
    # Get a unique color list, and use it to plot input dimensions
    colors = list(m_colors.TABLEAU_COLORS.values())
    handles = []
    labels = ['x pos', 'y pos', 'x vel', 'y vel']
    for i in range(x.shape[1]):
        if len(colors) < x.shape[1]:
            raise ValueError('need at least as many colors as channels')
        line, = ax.plot(x[0, i], color=colors[i % len(colors)], alpha=0.4, label=labels[i])
        # Add the legend entry for the first instance of each unique channel
        handles.append(line)

    time_diff = output_signal.shape[1] - x.shape[2]

    # Plot the output signal with a thicker line on top (foreground), in black
    conv_line, = ax.plot(np.arange(-time_diff // 2, x.shape[-1] + time_diff // 2), output_signal.squeeze(),
                         color='black', linewidth=1, label='output')
    handles.append(conv_line)

    # Return the handles for the legend
    return handles


if __name__ == '__main__':
    main(42, use_cached_features=False) #9000 #2