"""
Interprets the kernels and classifier weights of the ROCKET model and its ridge classifier.

In its current state, this file provides a starting point: loading the best ROCKET model and its classifier and
performing inference on the validation (or train/test) set. This provides access to the ROCKET features, classifier
predictions, labels, and input data.
"""

import torch
from joblib import load
from torch import tensor
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from utils.misc import set_random_state
from utils.path import config_path, rocket_instances_path
from utils.interpretability import feature_detachment, select_optimal_model, retrain_optimal_model
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np

#def main(seed, correct_way=True):

set_random_state(1337)
correct_way = True

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
rocket = torch.load(rocket_instances_path.joinpath('rocket_1337.ckpt'))

# Load trained classifier
ridge_clf = load(rocket_instances_path.joinpath('rocket_1337_clf.pkl'))

# Batch is entire dataset
train_batch = next(iter(dm.train_dataloader()))
test_batch = next(iter(dm.test_dataloader()))
val_batch = next(iter(dm.val_dataloader()))

# Perform ROCKET transformation stage on train, val, and test data
#train_features = rocket(train_batch.x).numpy()
#test_features = rocket(test_batch.x).numpy()
#val_features = rocket(val_batch.x).numpy()

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
train_val_features = np.concatenate((train_features,val_features),axis=0)
train_val_labels = np.concatenate((train_labels,val_labels),axis=0)

# Compute balanced accuracy
train_score = balanced_accuracy_score(train_labels, train_pred)
test_score = balanced_accuracy_score(test_labels, test_pred)
val_score = balanced_accuracy_score(val_labels, val_pred)

if(correct_way):
    # CORRECT WAY (train and val are combined)
    percentage_vector, score_list_train, score_list_val, feature_importance_matrix = feature_detachment(ridge_clf, train_features, val_features, train_labels, val_labels)

    optimal_index, optimal_percentage = select_optimal_model(percentage_vector, score_list_val, score_list_val[0])

else:
    # INCORRECT WAY (train and val are not combined)
    percentage_vector, score_list_train, score_list_test, feature_importance_matrix = feature_detachment(ridge_clf, train_val_features, test_features, train_val_labels, test_labels)

    optimal_index, optimal_percentage = select_optimal_model(percentage_vector, score_list_test, score_list_test[0])

# call retrain optimal model
retrained_clf = retrain_optimal_model(feature_importance_matrix, train_val_features, test_features, train_val_labels, test_labels, train_score, test_score, ridge_clf.alpha, optimal_index)
# Print the optimal percentage of features in percentage style with 2 decimal places
print(' ')
print('Used percentage of features: ' + str(round(100*optimal_percentage,2)) + '%')

#if __name__ == '__main__':
    # 1337 is the seed used for the best performing rocket model.
#    seed = 1337
#    correct_way = False
#    main(seed, correct_way)
