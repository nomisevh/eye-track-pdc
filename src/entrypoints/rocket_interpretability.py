"""
Interprets the kernels and classifier weights of the ROCKET model and its ridge classifier.

This file loads the best ROCKET model and its classifier, and computes feature importance based on Sequential Feature
Detachment. The optimal selection of kernels is then analyzed based on the kernel dilation and length.
"""

import numpy as np
import torch
from joblib import load
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from utils.interpretability import feature_detachment, select_optimal_model, retrain_optimal_model
from utils.misc import set_random_state
from utils.path import config_path, rocket_instances_path


def main(seed, correct_way=True, use_cached_features=True):
    set_random_state(seed)

    # Load configs
    with open(config_path.joinpath('leif.yaml')) as reader:
        processor_config = load_yaml(reader, Loader=FullLoader)

    dm = KIDataModule(processor_config=processor_config,
                      bundle_as_experiments=False,
                      exclude=['vert'],
                      binary_classification=True,
                      batch_size=-1)
    dm.setup('fit')
    dm.setup('test')

    # Load best rocket model
    rocket = torch.load(rocket_instances_path.joinpath(f'rocket_{seed}.ckpt'))

    # Load trained classifier
    ridge_clf = load(rocket_instances_path.joinpath(f'rocket_{seed}_clf.pkl'))

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
    except FileNotFoundError:
        # Perform ROCKET transformation stage on train, val, and test data
        train_features = rocket(train_batch.x).numpy()
        test_features = rocket(test_batch.x).numpy()
        val_features = rocket(val_batch.x).numpy()
        torch.save(train_features, features_filename('train'))
        torch.save(test_features, features_filename('test'))
        torch.save(val_features, features_filename('val'))

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
    retrained_clf = retrain_optimal_model(feature_importance_matrix, train_val_features, test_features,
                                          train_val_labels, test_labels, train_score, test_score, ridge_clf.alpha,
                                          optimal_index)

    # Print the optimal percentage of features in percentage style with 2 decimal places
    print(' ')
    print('Used percentage of features: ' + str(round(100 * optimal_percentage, 2)) + '%')

    compare_kernel_distributions(rocket.convs, feature_selection_matrix[optimal_index])


def compare_kernel_distributions(kernels, feature_selection):
    # Reshape by kernel, there are 2 features per kernel
    feature_selection = feature_selection.reshape((-1, 2))  # (num_kernels, 2)
    # Keep all kernels where at least one attribute was selected
    keep_mask = feature_selection.any(axis=1)
    kept_kernels = nn.ModuleList()
    discarded_kernels = nn.ModuleList()
    for kernel, keep in zip(kernels, keep_mask):
        if keep:
            kept_kernels.append(kernel)
        else:
            discarded_kernels.append(kernel)

    # Histogram the distribution of kernel dilation
    kept_dilation = [k.dilation[0] for k in kept_kernels]
    all_dilation = [k.dilation[0] for k in kernels]
    hist_kernel_property(kept_dilation, all_dilation, 'Dilation')

    # Do the same for kernel length
    kept_len = [k.weight.shape[-1] for k in kept_kernels]
    all_len = [k.weight.shape[-1] for k in kernels]
    hist_kernel_property(kept_len, all_len, 'Length')

    # And the bias
    kept_bias = [k.bias.item() for k in kept_kernels]
    all_bias = [k.bias.item() for k in kernels]
    hist_kernel_property(kept_bias, all_bias, 'Bias')

    # Receptive field
    kept_rf = [k.dilation[0] * k.weight.shape[-1] for k in kept_kernels]
    all_rf = [k.dilation[0] * k.weight.shape[-1] for k in kernels]
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


if __name__ == '__main__':
    main(1337)
