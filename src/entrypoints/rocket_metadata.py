import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import tensor
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
from utils.misc import set_random_state
from utils.path import ki_data_path, rocket_instances_path, config_path
from utils.visualize import plot_latent_neighborhood

SEED = 2


def main():
    set_random_state(SEED)

    with open(config_path.joinpath('leif.yaml')) as reader:
        processor_config = load_yaml(reader, Loader=FullLoader)

    dm = KIDataModule(processor_config=processor_config,
                      bundle_as_experiments=False,
                      exclude=['vert'],
                      binary_classification=True,
                      batch_size=-1)
    dm.setup('test')

    # Batch is entire dataset
    test_batch = next(iter(dm.test_dataloader()))

    # Initialize rocket with saved weights
    rocket = torch.load(rocket_instances_path.joinpath(f'pruned_rocket_{SEED}.ckpt'))

    # Initialize the classifier with saved weights
    with open(rocket_instances_path.joinpath(f'pruned_rocket_clf_{SEED}.pkl'), 'rb') as reader:
        ridge_clf = pickle.load(reader)

    test_features = rocket(test_batch.x)
    scores = ridge_clf.decision_function(test_features)
    preds = ridge_clf.predict(test_features)
    # probs = np.exp(scores) / np.sum(np.exp(scores))
    scores_normalized = scores - scores.mean() / scores.std()

    df = pd.read_excel(ki_data_path.joinpath('age_ID_table.xlsx'))

    # Remove one datapoint that is known to be inaccurate (124 y/o)
    df.drop(df[df.ID == 43].index, inplace=True)

    plot_latent_neighborhood(test_features, test_batch, dm.class_names(), show=True, metadata=df, key='Age')

    compute_metadata_correlation(preds, scores_normalized, test_batch.z, metadata=df, labels=test_batch.y)

    # The RidgeClassifier maps the targets to {-1, 1}, but our labels are {0, 1}
    preds[preds < 0] = 0

    trial_probs = tensor(preds)

    # evaluate(test_batch, trial_probs, test_features, dm.class_names(), model_name='ROCKET')


# Computes the correlation between the prediction scores and the various series in the metadata dataframe.
def compute_metadata_correlation(predictions, scores, subject_ids, metadata, labels):
    # Create mask for all subjects in the data that has metadata.
    has_metadata = np.isin(subject_ids, metadata['ID'])
    # Ignore datapoints for which we don't have metadata
    labels = labels[has_metadata]
    subject_ids = subject_ids[has_metadata]
    scores = scores[has_metadata]
    predictions = predictions[has_metadata]

    # Reorder and repeat rows in df based on subject IDs for the datapoints above
    reordered_df = pd.concat([metadata[metadata.ID == s_id] for s_id in subject_ids.tolist()])

    # Compute correlation for each series in metadata
    correlations = {
        'Age': np.corrcoef(scores, reordered_df['Age']),
        'Labels': np.corrcoef(labels, reordered_df['Age']),
        'Predictions': np.corrcoef(predictions, reordered_df['Age']),
    }

    fig, ax = plt.subplots()

    ax.scatter(labels, reordered_df['Age'], c=predictions == labels.numpy)

    return correlations


if __name__ == '__main__':
    main()
