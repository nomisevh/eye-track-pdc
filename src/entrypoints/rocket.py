import pickle

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier,RidgeClassifierCV
from torch import tensor
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from models.rocket import ROCKET
from utils.eval import evaluate
from utils.misc import set_random_state
from utils.path import config_path, rocket_instances_path


def main(seed, use_pruned_model=False):
    set_random_state(seed)
    print('-----------------------------------')
    print(f'Running with seed: {seed}')

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

    # Initialize Rocket
    if use_pruned_model:
        # The loaded rocket model has the statistics of all features when normalizing, keep it that way.
        rocket = torch.load(rocket_instances_path.joinpath(f'pruned_rocket_{seed}.ckpt'))
    else:
        rocket = ROCKET(c_in=dm.train_ds.x.shape[1],
                        seq_len=dm.train_ds.x.shape[2],
                        **rocket_config)

    # Batch is entire dataset
    train_batch = next(iter(dm.train_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))

    # Added since no val test is being used for training, we will add it to the training data
    val_batch = next(iter(dm.val_dataloader()))


    # Perform ROCKET transformation stage on train and test data
    rocket.train = True
    # I cahnged this line to concatenate the train and val data
    train_val_features = rocket(torch.cat((train_batch.x, val_batch.x), 0)).numpy()
    rocket.train = False
    test_features = rocket(test_batch.x)

    #train_features = rocket(train_batch.x)
    #val_features = rocket(val_batch.x)

    #print(f'Train features shape: {train_features.shape}')

    print(f'Train_val features shape: {train_val_features.shape}')

    print(f'Percentage of used features: {100 * train_val_features.shape[1]/20000}%')
          
    #train_labels = train_batch.y
    #val_labels = val_batch.y

    # Combined labels for train and val
    train_val_labels = torch.cat((train_batch.y, val_batch.y), 0)
    

    # Initialize Classifiers
    if use_pruned_model:
        #with open(rocket_instances_path.joinpath(f'pruned_rocket_clf_{seed}.pkl'), 'rb') as reader:
        #    ridge_clf = pickle.load(reader)

        #ridge_clf = RidgeClassifierCV(alphas=np.logspace(0,3,10))
        

        ridge_clf = RidgeClassifier(alpha=1e3)
        
        # list_alphas = np.logspace(1,4,20)
        # best_alpha = find_best_alpha(list_alphas, train_features, train_labels, val_features, val_labels)
        # ridge_clf = RidgeClassifier(alpha=best_alpha)

    else:
        ridge_clf = RidgeClassifier(alpha=1e3)
        #ridge_clf = RidgeClassifierCV(alphas=np.logspace(-10,10,20))


    # Fit classifier to the rocket features
    ridge_clf.fit(train_val_features, train_val_labels)
    #print(f'Alpha used: {ridge_clf.alpha_}')
    #ridge_clf.fit(train_features, train_batch.y)

    # Make predictions on test set
    test_pred = ridge_clf.predict(test_features)

    # The RidgeClassifier maps the targets to {-1, 1}, but our labels are {0, 1}
    test_pred[test_pred < 0] = 0

    test_trial_probs = tensor(test_pred)

    scores = ridge_clf.decision_function(test_features)
    sorted_indices = scores.argsort()

    # Initialize an array of the same shape as arr to hold the ranks
    ranks = np.empty_like(sorted_indices, dtype=float)

    # Assign ranks to the sorted elements (starting from 1)
    ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)

    sorted_scores = ranks / ranks.max()

    evaluate(test_batch, test_trial_probs, test_features, dm.class_names(), model_name='ROCKET',
             test_trial_scores=sorted_scores)

from sklearn.metrics import accuracy_score
from torchmetrics.functional.classification import multiclass_f1_score

def find_best_alpha(alphas, X_train, y_train, X_val, y_val):
    best_alpha = None
    best_score = 0
    total_iterations = len(alphas)
    num_iteration = 0
    for alpha in alphas:
        print(f'Itieration {num_iteration} out of {total_iterations}')
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        #score = accuracy_score(y_val, y_pred)
        score = multiclass_f1_score(tensor(y_pred), tensor(y_val), num_classes=2)

        if score > best_score:
            best_score = score
            best_alpha = alpha
        
        num_iteration += 1

    print(f'Best alpha: {best_alpha}, Best score: {best_score}')
    return best_alpha

if __name__ == '__main__':
    #main(2, use_pruned_model=True)

     for seed in [42, 1337, 9000, 1, 2]:
         main(seed, use_pruned_model=True)
