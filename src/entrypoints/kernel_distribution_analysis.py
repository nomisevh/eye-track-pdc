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
import os

from joblib import load
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score
from torch import nn, tensor
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from models.rocket import dissected_forward, ROCKET
from utils.interpretability import feature_detachment, select_optimal_model, retrain_optimal_model
from utils.misc import set_random_state
from utils.path import config_path, rocket_instances_path, figure_path, ki_data_tmp_path


def main():

    seed_list = [42,1337,9000,1,2]  

    all_kernels_list = []
    kept_kernels_list = []

    for seed in seed_list:
        # Load kernels
        with open(rocket_instances_path.joinpath(f'all_kernels_{seed}.pkl'), 'rb') as reader:
            all_kernels = pickle.load(reader)
        with open(rocket_instances_path.joinpath(f'kept_kernels_{seed}.pkl'), 'rb') as reader:
            kept_kernels = pickle.load(reader)
        
        all_kernels_list.append(all_kernels)
        kept_kernels_list.append(kept_kernels)
    
    kept_dilation_list = []
    all_dilation_list = []

    hist_all_dilation_list = []
    hist_kept_dilation_list = []

    relative_ketp_dilation = []

    kept_len_list = []
    all_len_list = []

    kept_receptive_field_list = []
    all_receptive_field_list = []

    hist_all_rf_list = []
    hist_kept_rf_list = []

    relative_ketp_rf = []

    for i in range(len(seed_list)):
        kept_kernels = kept_kernels_list[i]
        all_kernels = all_kernels_list[i]

        kept_dilation = [k.dilation[0] for k in kept_kernels]
        all_dilation = [k.dilation[0] for k in all_kernels]
        kept_dilation_list.append(kept_dilation)
        all_dilation_list.append(all_dilation)

        bins_dilation = np.arange(1, 71) - 0.5

        # Compute the amount of kernels with a dilation that falls into the ranges determined by the bins
        hist_all_dilation, _ = np.histogram(all_dilation, bins=bins_dilation)
        hist_kept_dilation, _ = np.histogram(kept_dilation, bins=bins_dilation)

        # Normalize the histograms
        hist_all_dilation = hist_all_dilation / len(all_dilation)
        hist_kept_dilation = hist_kept_dilation / len(kept_dilation)

        hist_all_dilation_list.append(hist_all_dilation)
        hist_kept_dilation_list.append(hist_kept_dilation)

        # Percentual change in the amount of kernels with a dilation that falls into the ranges determined by the bins
        percentual_change_dilation = (hist_kept_dilation - hist_all_dilation) / hist_all_dilation
        relative_ketp_dilation.append(percentual_change_dilation)

        kept_len = [k.weight.shape[-1] for k in kept_kernels]
        all_len = [k.weight.shape[-1] for k in all_kernels]
        kept_len_list.append(kept_len)
        all_len_list.append(all_len)

        kept_rf = [k.dilation[0] * k.weight.shape[-1] for k in kept_kernels]
        all_rf = [k.dilation[0] * k.weight.shape[-1] for k in all_kernels]
        kept_receptive_field_list.append(kept_rf)
        all_receptive_field_list.append(all_rf)

        # Compute the amount of kernels with a receptive field that falls into the ranges determined by the bins
        bins = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]

        #bins = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504]

        hist_all_rf, _ = np.histogram(all_rf, bins=bins)
        hist_kept_rf, _ = np.histogram(kept_rf, bins=bins)

        # Normalize the histograms
        hist_all_rf = hist_all_rf / len(all_rf)
        hist_kept_rf = hist_kept_rf / len(kept_rf)

        hist_all_rf_list.append(hist_all_rf)
        hist_kept_rf_list.append(hist_kept_rf)

        # Percentual change in the amount of kernels with a receptive field that falls into the ranges determined by the bins
        percentual_change_rf = (hist_kept_rf - hist_all_rf) / hist_all_rf
        relative_ketp_rf.append(percentual_change_rf)

        if i == 0:
            all_rf_together = all_rf
            kept_rf_together = kept_rf

            all_dilation = all_dilation
            kept_dilation = kept_dilation

        else:
            all_rf_together = all_rf_together + all_rf
            kept_rf_together = kept_rf_together + kept_rf

            all_dilation = all_dilation + all_dilation
            kept_dilation = kept_dilation + kept_dilation

    max_dilation_value = np.max(all_dilation_list)
    print(f"Max dilation value: {max_dilation_value}")

    sampling_frequency = 300 # Hz

    list_dilation_time_rf = np.asarray(bins[1:]) / sampling_frequency # seconds
    list_dilation_freq_rf = 1 / np.array(list_dilation_time_rf) # Hz

    dilation_values_list = bins_dilation[1:]-0.5
    dilation_frequency_list = sampling_frequency / np.array(dilation_values_list)

    # Plot all dilations histogram
    plt.figure()
    plt.hist([all_dilation, kept_dilation], bins=np.arange(1, max_dilation_value) - 0.5, label=['All Kernels', 'Kept Kernels'], density=True)
    plt.legend(loc='upper right')
    plt.title("Receptive Field")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.xlim(0, 20.5)
    plt.show()

    # Plot mean and std of the histograms of all dilations and kpt kernels for each seed
    plt.figure(figsize=(5, 4))
    mean = 100*np.mean(hist_all_dilation_list, axis=0)
    std = 100*np.std(hist_all_dilation_list, axis=0)
    plt.bar(np.asarray(dilation_values_list)-0.15, mean, yerr=std, label='Initial Distribution (pre-pruning)',width=0.3)
    mean_kept = 100*np.mean(hist_kept_dilation_list, axis=0)
    std_kept = 100*np.std(hist_kept_dilation_list, axis=0)
    plt.bar(np.asarray(dilation_values_list)+0.15, mean_kept, yerr=std_kept, label='Final Distribution (post-pruning)',width=0.3)
    plt.legend(loc='upper right')
    plt.xticks(dilation_values_list)
    plt.xlim(0.5, 12.5)
    plt.ylim(0, 25)
    plt.xlabel("Dilation Value")
    plt.ylabel("Percentage of Total Kernels (%)")
    plt.savefig(figure_path.joinpath('dilation_distribution.pdf'), format='pdf', dpi=1200)
    plt.show()

    


    # plt.figure()
    # plt.hist([all_rf_together, kept_rf_together], bins=bins, label=['All Kernels', 'Kept Kernels'], density=True)
    # plt.legend(loc='upper right')
    # plt.title("Receptive Field")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.show()

    # Total number of kept kernels
    print(f"Total number of kept kernels: {len(kept_rf_together)}")
    # Total number of all kernels
    print(f"Total number of all kernels: {len(all_rf_together)}")
    # Proportion of kept kernels
    print(f"Proportion of kept kernels: {len(kept_rf_together) / len(all_rf_together)}")


    # plt.figure()
    # plt.scatter(bins[1:], np.mean(relative_ketp_rf, axis=0))
    # plt.legend(loc='upper right')
    # plt.title("Receptive Field")
    # # Plot horizontal line at 0.0
    # plt.axhline(y=0.0, color='r', linestyle='--')
    # plt.xlabel("Value")
    # plt.ylabel("Percentual Change")
    # plt.show()

    # # Plot the percentual change for each seed
    # plt.figure()
    # for i in range(len(seed_list)):
    #     plt.scatter(bins[1:], relative_ketp_rf[i], label=f'Seed {seed_list[i]}')
    # plt.legend(loc='upper left')
    # plt.title("Receptive Field")
    # # Plot horizontal line at 0.0
    # plt.axhline(y=0.0, color='r', linestyle='--')
    # plt.xlabel("Value")
    # plt.ylabel("Percentual Change")
    # plt.show()

    # # Plot mean and standard deviation of the percentual change for each bin
    # plt.figure()
    # mean = np.mean(relative_ketp_rf, axis=0)
    # std = np.std(relative_ketp_rf, axis=0)
    # plt.errorbar(list_dilation_freq_rf, mean, yerr=std, fmt='o')
    # plt.legend(loc='upper right')
    # plt.title("Receptive Field")
    # # Plot horizontal line at 0.0
    # plt.axhline(y=0.0, color='r', linestyle='--')
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Percentual Change")
    # plt.show()

    # Plot mean and standard deviation of the percentual change for each bin
    plt.figure()
    mean = 100*np.mean(relative_ketp_dilation, axis=0)
    std = 100*np.std(relative_ketp_dilation, axis=0)
    plt.errorbar(dilation_values_list, mean, yerr=std, fmt='o')
    plt.legend(loc='upper right')
    plt.title("Receptive Field")
    # Define range of x-axis
    plt.xlim(0, 20.5)
    # Define range of y-axis
    plt.ylim(-50, 50)
    # Plot horizontal line at 0.0
    plt.axhline(y=0.0, color='r', linestyle='--')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Percentual Change")
    plt.show()

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



if __name__ == '__main__':
    main() #9000 #2