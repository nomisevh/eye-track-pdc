from sklearn.linear_model import RidgeClassifier
from torch import tensor
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from models.rocket import ROCKET
from utils.const import SEED
#from utils.eval import evaluate
from utils.numeric_eval import evaluate
from utils.misc import set_random_state
from utils.path import config_path,ki_data_tmp_path,figure_path
from utils.interpretability import compute_average_power_spectrum, plot_power_spectrum
from utils.ki import SAMPLE_RATE

import numpy as np
import torch

import os
import sys

#def main(seed):
seed = 1337
set_random_state(seed)

filter_order = 8

list_cutoff_freq = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60]

#list_cutoff_freq = [2,140]

list_trial_acc = []
list_subject_acc = []
list_trial_uf1 = []
list_subject_uf1 = []
list_attribute_power = []
list_avg_power_spectrum = []
list_center_freq = []
list_nans_train = []
list_nans_test = []

old_stdout = sys.stdout # backup current stdout

for cutoff_freq in list_cutoff_freq:

    # Pint cuttoff frequency
    print(f"Cuttoff frequency: {cutoff_freq}")

    # Redirect stdout to devnull to suppress output
    sys.stdout = open(os.devnull, "w")

    # Load configs
    with open(config_path.joinpath('leif.yaml')) as reader:
        processor_config = load_yaml(reader, Loader=FullLoader)
    with open(config_path.joinpath('rocket.yaml')) as reader:
        rocket_config = load_yaml(reader, Loader=FullLoader)

    processor_config['frequency_filter']['cutoff'] = cutoff_freq
    processor_config['frequency_filter']['order'] = filter_order

    # Remove tmp files from ki_data_tmp_path folder
    file_to_remove_test = ki_data_tmp_path.joinpath('ki-HC,PD_OFF,PD_ON-seg-test.pth')
    file_to_remove_train = ki_data_tmp_path.joinpath('ki-HC,PD_OFF,PD_ON-seg-train.pth')

    if file_to_remove_test.exists():
        os.remove(file_to_remove_test)

    if file_to_remove_train.exists():
        os.remove(file_to_remove_train)

    dm = KIDataModule(processor_config=processor_config,
                        bundle_as_experiments=False,
                        exclude=['vert'],
                        binary_classification=True,
                        batch_size=-1)
    dm.setup('fit')
    dm.setup('test')

    # Initialize Rocket
    rocket = ROCKET(c_in=dm.train_ds.x.shape[1],
                    seq_len=dm.train_ds.x.shape[2],
                    **rocket_config)

    # Initialize Classifiers
    ridge_clf = RidgeClassifier(alpha=1e4, random_state=SEED)

    # Batch is entire dataset
    train_batch = next(iter(dm.train_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))

    # Perform ROCKET transformation stage on train and test data
    train_features = rocket(train_batch.x)
    test_features = rocket(test_batch.x)

    # Count nans in train and test pytorch tensors
    nans_train = torch.isnan(train_features).sum().numpy().item()
    nans_test = torch.isnan(test_features).sum().numpy().item()

    # Replace nans with zeros
    train_features = torch.nan_to_num(train_features)
    test_features = torch.nan_to_num(test_features)

    # Fit classifier to the rocket features
    ridge_clf.fit(train_features, train_batch.y)

    # Make predictions on test set
    test_pred = ridge_clf.predict(test_features)

    # The RidgeClassifier maps the targets to {-1, 1}, but our labels are {0, 1}
    test_pred[test_pred < 0] = 0

    test_trial_probs = tensor(test_pred)

    result_dict = evaluate(test_batch, test_trial_probs, test_features, dm.class_names(), model_name='ROCKET')

    list_of_arrays = [t.x[0].numpy() for t in dm.train_ds if t.y == 1]

    avg_power_spectrum, center_freq = compute_average_power_spectrum(list_of_arrays, SAMPLE_RATE, 30)

    # Compute for Dummy predictor (predicting always predominant class)
    dummy_test_trial_probs = evaluate(test_batch,tensor(np.ones_like(test_pred)),test_features, dm.class_names(), model_name='ROCKET')

    #plot_power_spectrum(center_freq, avg_power_spectrum)

    # Unpack result dictionary and append
    list_trial_acc.append(result_dict['test_trial_acc'])
    list_subject_acc.append(result_dict['test_subject_acc_best'])
    list_trial_uf1.append(result_dict['test_trial_uf1_score'])
    list_subject_uf1.append(result_dict['test_subject_uf1_score_best'])
    list_attribute_power.append(result_dict['attribute_power'])

    list_avg_power_spectrum.append(avg_power_spectrum)
    list_center_freq.append(center_freq)

    list_nans_train.append(nans_train)
    list_nans_test.append(nans_test)

    
    sys.stdout = old_stdout # reset old stdout
    
    # Print test subject uf1
    print(f"Test subject uF1: {result_dict['test_subject_uf1_score_best']:.4f}")

    # Print number of nans in test set
    print(f"Number of nans in test set: {nans_test}")

# Define path to save results
results_file_path = figure_path.joinpath('dict_results.pickle')

# Create a results dictionary with all lists 
dict_results = {'list_cutoff_freq': list_cutoff_freq,
                'list_trial_acc': list_trial_acc,
                'list_subject_acc': list_subject_acc,
                'list_trial_uf1': list_trial_uf1,
                'list_subject_uf1': list_subject_uf1,
                'list_attribute_power': list_attribute_power,
                'list_avg_power_spectrum': list_avg_power_spectrum,
                'list_center_freq': list_center_freq,
                'list_nans_train': list_nans_train,
                'list_nans_test': list_nans_test}

# Save dictionary in pickle file
import pickle

with open(results_file_path, 'wb') as handle:
    pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

import matplotlib.pyplot as plt


## FIGURE 1 -------------------------------------------------------------------

dummy_trial_uF1 = dummy_test_trial_probs['test_trial_uf1_score'].numpy()
dummy_subject_uF1 = dummy_test_trial_probs['test_subject_uf1_score_best'].numpy()

# Set figure size for one column article
plt.figure(figsize=(5, 4))

# Plot cutoff frequency vs. test subject uf1
plt.plot(list_cutoff_freq, list_subject_uf1, 'o-', label='Subject uF1 Score', color='C0')
plt.plot(list_cutoff_freq, list_trial_uf1, 'o-', label='Trial uF1 Score', color='C1')
plt.xlabel('Cutoff frequency [Hz]')
plt.ylabel('Test Classification uF1 Score')

# Fill region between 4 and 7 Hz with red
plt.fill_between([4,7], 0.3, 1, facecolor='C2', alpha=0.3, label='Tremor Frequency Range')

# Fill region between 8 and 14 Hz with red
plt.fill_between([8,14], 0.3, 1, facecolor='C4', alpha=0.3, label='First Harmonic Range')

# Plot dummy uF1
plt.plot(list_cutoff_freq, [dummy_subject_uF1 for i in range(len(list_cutoff_freq))], '--', color='C0')#, label='Subject uF1 - Dummy')

plt.plot(list_cutoff_freq, [dummy_trial_uF1 for i in range(len(list_cutoff_freq))], '--', color='C1')#, label='Trial uF1 - Dummy')

plt.legend()
plt.grid()
plt.tight_layout()

# Define path to save figure
figure_file_path = figure_path.joinpath('cutoff_freq_vs_test_subject_uf1.pdf')

# Save figure as svg
plt.savefig(figure_file_path, format='pdf')

# Show figure
plt.show()

## FIGURE 2 -------------------------------------------------------------------

# Plot all power spectra with color to indicate freq value

# Create figure
fig, ax = plt.subplots()

# Create color map
from matplotlib import cm
cmap = cm.get_cmap('viridis')

# Create colorbar
from matplotlib import colors
norm = colors.Normalize(vmin=min(list_cutoff_freq), vmax=max(list_cutoff_freq))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Cutoff frequency [Hz]')

# Plot all power spectra
for i in range(len(list_avg_power_spectrum)):
    ax.plot(list_center_freq[i], list_avg_power_spectrum[i], color=cmap(norm(list_cutoff_freq[i])), label=f'{list_cutoff_freq[i]} Hz')

# Fill region between 4 and 7 Hz with red
ax.fill_between([4,7], 0, 0.3, facecolor='red', alpha=0.2, label='Tremor range')

# Fill region between 8 and 14 Hz with red
ax.fill_between([8,14], 0, 0.3, facecolor='orange', alpha=0.2, label='First Harmonic Range')

# Set x and y labels
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Power')

# Set title
ax.set_title('Average power spectrum for all subjects')

# Set grid
ax.grid()
plt.tight_layout()

# Save figure as svg
figure_file_path = figure_path.joinpath('power_spectrum_all_subjects.svg')
plt.savefig(figure_file_path, format='svg')

# Show figure
plt.show()


## FIGURE 3 -------------------------------------------------------------------
# Same as Figure 1 but with two y-axes

# Create figure
fig, ax1 = plt.subplots()

# Plot cutoff frequency vs. test subject uf1
ax1.plot(list_cutoff_freq, list_subject_uf1, 'o-', label='Test subject uF1')
ax1.set_xlabel('Cutoff frequency')
ax1.set_ylabel('Test subject uF1')

# Fill region between 4 and 7 Hz with red
ax1.fill_between([4,7], 0.5, 1, facecolor='red', alpha=0.2, label='Tremor range')

# Fill region between 8 and 14 Hz with red
ax1.fill_between([8,14], 0.5, 1, facecolor='orange', alpha=0.2, label='First Harmonic Range')

# Set grid
ax1.grid()

# Create second y-axis
ax2 = ax1.twinx()

# Power between 3 and 7 Hz
main_tremor_power_list = []
for i in range(len(list_avg_power_spectrum)):
    avg_power_spectrum = list_avg_power_spectrum[i]
    main_indices = np.where((center_freq >= 3) & (center_freq < 7))
    main_tremor_power = np.sum(avg_power_spectrum[main_indices])
    main_tremor_power_list.append(main_tremor_power)

# Power between 8 and 14 Hz
first_harmonic_power_list = []
for i in range(len(list_avg_power_spectrum)):
    avg_power_spectrum = list_avg_power_spectrum[i]
    first_harmonic_indices = np.where((center_freq >= 8) & (center_freq < 14))
    first_harmonic_power = np.sum(avg_power_spectrum[first_harmonic_indices])
    first_harmonic_power_list.append(first_harmonic_power)


# Plot power
ax2.plot(list_cutoff_freq, [p + 0.5 for p in main_tremor_power_list], '--', label='Fundamental Frequency power (3Hz-7Hz)', color='red')
ax2.plot(list_cutoff_freq, [p + 0.5 for p in first_harmonic_power_list], '--', label='First harmonic power (7Hz-14Hz)', color='orange')
ax2.set_ylabel('Proportion of Spectrum Power')

ax2.set_yticks(ax1.get_yticks())
ax2.set_ylim(ax1.get_ylim())
ax2.set_yticklabels([ '','0', '0.1', '0.2','0.3','0.4','0.5',''])

# Set legend
#ax1.legend(loc='upper left')
#ax2.legend(loc='upper right')

# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.tight_layout()

# Save figure as svg
figure_file_path = figure_path.joinpath('cutoff_freq_vs_test_subject_uf1_two_y_axes.svg')
plt.savefig(figure_file_path, format='svg')

# Show figure
plt.show()


#if __name__ == '__main__':
#    main(1337)

    # for seed in [42, 1337, 9000, 1, 2]:
    #     main(seed)
