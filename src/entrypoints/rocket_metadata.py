import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import tensor
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
from utils.misc import set_random_state
from utils.path import ki_data_path, rocket_instances_path, config_path, figure_path
from utils.visualize import plot_latent_neighborhood

from scipy import stats

SEED = 9000 # 2 # 9000 # 42


def main():
    set_random_state(SEED)

    with open(config_path.joinpath('leif.yaml')) as reader:
        processor_config = load_yaml(reader, Loader=FullLoader)

    dm = KIDataModule(processor_config=processor_config,
                      bundle_as_experiments=False,
                      exclude=['vert'],
                      binary_classification=True,
                      val_size=0.2, # Change to zero for having all the training data in the plot (0.2 default)
                      batch_size=-1)
    
    # Select the set to visualize 
    selected_set = 'test' # 'test' 'train' 'all'

    # Select the feature to visualize
    selected_feature = ['Duration'] #['Age'] ['UPDRS_ON'] ['UPDRS_OFF'] ['MoCA'] ['FAB'] ['Duration'] ['Subtype']

    if selected_set == 'train':
        dm.setup('fit')
        # Batch is entire dataset
        selected_batch = next(iter(dm.train_dataloader()))
    elif selected_set == 'test':
        dm.setup('test')
        # Batch is entire dataset
        selected_batch = next(iter(dm.test_dataloader()))
    elif selected_set == 'all':
        dm.setup('fit')
        selected_batch = next(iter(dm.train_dataloader()))
        dm.setup('test')
        # Batch is entire dataset
        selected_batch = selected_batch + next(iter(dm.test_dataloader()))
    #print(type(selected_batch))

    # Initialize rocket with saved weights
    rocket = torch.load(rocket_instances_path.joinpath(f'pruned_rocket_{SEED}.ckpt'))

    # Load the normalization parameters
    with open(rocket_instances_path.joinpath(f'pruned_rocket_normalization_{SEED}.pkl'), 'rb') as reader:
        rocket_params = pickle.load(reader)
        rocket.mean = rocket_params['mean']
        rocket.std = rocket_params['std']
    rocket.train = False

    # Initialize the classifier with saved weights
    with open(rocket_instances_path.joinpath(f'pruned_rocket_clf_{SEED}.pkl'), 'rb') as reader:
        ridge_clf = pickle.load(reader)

    selected_features = rocket(selected_batch.x)
    scores = ridge_clf.decision_function(selected_features)
    preds = ridge_clf.predict(selected_features)
    # probs = np.exp(scores) / np.sum(np.exp(scores))
    scores_normalized = (scores - scores.mean()) / scores.std()

    #df = pd.read_excel(ki_data_path.joinpath('age_ID_table.xlsx'))
    df = pd.read_excel(ki_data_path.joinpath('metadata_ID_table.xlsx'))


    # Convert column ['Subtype'] to a categorical variable with values 0 when 'rigidity', 1 when 'mixed' and 2 when 'tremor'

    # Create a dictionary to map the values to integers
    mapping = {'rigid': 1, 'mixed': 2, 'tremor': 3}

    # Apply the mapping to the column ['Subtype']
    df['Subtype'] = df['Subtype'].map(mapping)

    # Remove one datapoint that is known to be inaccurate (124 y/o)
    df.drop(df[df.ID == 43].index, inplace=True)

    # remove rows with missing values in selected_feature
    df = df.dropna(subset=selected_feature)

    plot_latent_neighborhood(selected_features, selected_batch, dm.class_names(), filename='age_matched' + selected_set+'_plot_latent_space_'+selected_feature[0], show=True, metadata=df, key=selected_feature[0],scores=list(scores_normalized))

    plt.scatter(selected_batch.y, scores_normalized, c=preds == selected_batch.y.numpy())
    plt.show()

    #print(scores_normalized.mean())
    #print(type(list(scores_normalized)))
    #print(type(scores_normalized[0]))

    correlations = compute_metadata_correlation(preds, scores_normalized, selected_batch.z, metadata=df, labels=selected_batch.y, selected_feature=selected_feature[0])

    # Print correlation coefficients of all keys in correlations
    for key in correlations:
        try:
            print(f'Correlation between {selected_feature[0]} and {key}: {correlations[key][0, 1]}')
        except:
            print(f'Correlation between {selected_feature[0]} and {key}: {correlations[key].correlation}') 
            print(f'p-value: {correlations[key].pvalue}')
            print(' ')

    if selected_feature[0] == 'Age':        
        # Compute KS test for the age feature
        ks_test = compute_ks_test_age(selected_batch.z, df, 'age_matched' + selected_set+'_age_distribution')

        # Print KS test results (both the statistic and the p-value)
        print('Age distribution in HC and PD groups:')
        print(f'KS test statistic: {ks_test.statistic}')
        print(f'KS test p-value: {ks_test.pvalue}')

        # Determine if we can reject the null hypothesis
        if ks_test.pvalue < 0.05:
            print('The null hypothesis can be rejected. The two distributions are different.')
        else:
            print('The null hypothesis cannot be rejected. The two distributions may be the same.')

    correlations_medication = compute_correlation_medication(selected_batch.g, scores_normalized, selected_batch.z, metadata=df, labels=selected_batch.y)

    # Print the correlation coefficients of the medication status
    print(f'Patient Correlation between score and medication status: {correlations_medication["Scores"].correlation}')
    print(f'p-value: {correlations_medication["Scores"].pvalue}')
    print(' ')

    # The RidgeClassifier maps the targets to {-1, 1}, but our labels are {0, 1}
    #preds[preds < 0] = 0
    # trial_probs = tensor(preds)
    # evaluate(selected_batch, trial_probs, selected_features, dm.class_names(), model_name='ROCKET')

    correlations = compute_metadata_correlation_patient(preds, scores_normalized, selected_batch.z, metadata=df, labels=selected_batch.y, selected_feature=selected_feature[0])
    # Print correlation coefficients of all keys in correlations
    for key in correlations:
        try:
            print(f'Patient Correlation between {selected_feature[0]} and {key}: {correlations[key][0, 1]}')
        except:
            print(f'Patient Correlation between {selected_feature[0]} and {key}: {correlations[key].correlation}') 
            print(f'p-value: {correlations[key].pvalue}')
            print(' ')

    return selected_batch


# Computes the correlation between the prediction scores and the various series in the metadata dataframe.
def compute_metadata_correlation(predictions, scores, subject_ids, metadata, labels, selected_feature):
    # Create mask for all subjects in the data that has metadata.
    has_metadata = np.isin(subject_ids, metadata['ID'])
    # Ignore datapoints for which we don't have metadata
    labels = labels[has_metadata]
    subject_ids = subject_ids[has_metadata]
    scores = scores[has_metadata]
    predictions = predictions[has_metadata]

    # Print number of trials
    print(f'Number of trials: {len(scores)}')

    # Reorder and repeat rows in df based on subject IDs for the datapoints above
    reordered_df = pd.concat([metadata[metadata.ID == s_id] for s_id in subject_ids.tolist()])

    # Correlation with labels is only relevant when selected_feature is 'Age' (there are two classes)
    if selected_feature == 'Age':
        # Compute correlation for each series in metadata
        res = stats.spearmanr(scores, reordered_df[selected_feature], alternative='two-sided')
        correlations = {
            #'Scores': np.corrcoef(scores, reordered_df[selected_feature]),
            'Scores': res,
            'Labels': stats.spearmanr(labels, reordered_df[selected_feature]),
            'Predictions': stats.spearmanr(predictions, reordered_df[selected_feature]),
        }

        fig, ax = plt.subplots()
        ax.scatter(labels, reordered_df[selected_feature], c=predictions == labels.numpy)

    else:
        # Compute correlation for each series in metadata
        res = stats.spearmanr(scores, reordered_df[selected_feature], alternative='two-sided')
        correlations = {
            #'Scores': np.corrcoef(scores, reordered_df[selected_feature]),
            'Scores': res,
            'Predictions': stats.spearmanr(predictions, reordered_df[selected_feature])
        }
        

    return correlations

# Compute the correlation (only in the patient group) between the prediction scores and the medication status PDON or PDOFF (medication status come like {'HC': 0, 'PDOFF': 1, 'PDON': 2})
def compute_correlation_medication(medication_status, scores, subject_ids, metadata, labels):

    # Selec only PD subjects
    is_PD = medication_status != 0
    subject_ids = subject_ids[is_PD]
    scores = scores[is_PD]
    medication_status = medication_status[is_PD]
    labels = labels[is_PD]

    # Map, if PDOFF -> 1, PDON -> 0
    medication_status = np.where(medication_status == 1, 0, 1)

    # Compute correlation for each series in metadata
    res = stats.spearmanr(scores, medication_status, alternative='two-sided')
    correlations = {
        'Scores': res
    }

    return correlations


def compute_ks_test_age(subject_ids, metadata, figure_filename):

    # Create mask for all subjects in the data that has metadata.
    has_metadata = np.isin(subject_ids, metadata['ID'])
    # Ignore datapoints for which we don't have metadata
    subject_ids = subject_ids[has_metadata]

    # Reorder and repeat rows in df based on subject IDs for the datapoints above
    reordered_df = pd.concat([metadata[metadata.ID == s_id] for s_id in subject_ids.tolist()])

    # Create a new dataset with only the HC group
    df_HC = reordered_df[reordered_df['grupp']=='HC']
    # Now select only the first entrance belonging to each subject
    df_HC = df_HC.drop_duplicates(subset='ID', keep='first')

    # Print number of subjects in HC group
    print(f'Number of subjects in HC group: {len(df_HC)}')

    # Create a new dataset with only the PD group
    df_PD = reordered_df[reordered_df['grupp']=='PD']
    # Now select only the first entrance belonging to each subject
    df_PD = df_PD.drop_duplicates(subset='ID', keep='first')

    # Print number of subjects in PD group
    print(f'Number of subjects in PD group: {len(df_PD)}')


    # Compute a KS test to compare the distribution of ages feature in the HC and PD groups
    ks_test = stats.ks_2samp(df_HC['Age'], df_PD['Age'])

    # Plot histograms of the age feature for the HC and PD groups
    fig, ax = plt.subplots()

    # Plot both histograms with the same bins
    bins = np.linspace(30, 90, 20)
    ax.hist(df_HC['Age'], bins, alpha=0.4, label='HC')
    ax.hist(df_PD['Age'], bins, alpha=0.4, label='PD')

    ax.legend(loc='upper right')
    ax.set_title('Age distribution in HC and PD groups')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    plt.savefig(f'{figure_path.joinpath(figure_filename)}.svg', format='svg', dpi=1200)
    plt.show()

    return ks_test

# Computes the correlation between the prediction scores and the various series in the metadata dataframe, but it aggregates the data for each subject.
def compute_metadata_correlation_patient(predictions, scores, subject_ids, metadata, labels, selected_feature):
    # Create mask for all subjects in the data that has metadata.
    has_metadata = np.isin(subject_ids, metadata['ID'])
    # Ignore datapoints for which we don't have metadata
    labels = labels[has_metadata]
    subject_ids = subject_ids[has_metadata]
    scores = scores[has_metadata]
    predictions = predictions[has_metadata]

    # Reorder and repeat rows in df based on subject IDs for the datapoints above
    reordered_df = pd.concat([metadata[metadata.ID == s_id] for s_id in subject_ids.tolist()])

    # Aggregate scores for each subject
    scores_agg = []
    ids_list = []
    for s_id in subject_ids.unique():
        scores_agg.append(scores[subject_ids == s_id].mean())
        ids_list.append(int(s_id.numpy()))

    # Print number of Subjects
    print(f'Number of Subjects: {len(scores_agg)}')

    # Get the selected feature for each subject
    selected_feature_agg = []
    for s_id in ids_list:
        selected_feature_agg.append(reordered_df[reordered_df.ID == s_id][selected_feature].iloc[0])

    #print(scores_agg)
    #print(selected_feature_agg)
    #plt.scatter(scores_agg, selected_feature_agg)
    #plt.show()

    res = stats.spearmanr(scores_agg, selected_feature_agg, alternative='two-sided')
    correlations = {
        'Scores': res,
    }

    return correlations




if __name__ == '__main__':
    main()
