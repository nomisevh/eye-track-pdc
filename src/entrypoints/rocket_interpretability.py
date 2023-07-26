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


def main(seed):
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
    rocket = torch.load(rocket_instances_path.joinpath('rocket_1337.ckpt'))

    # Load trained classifier
    ridge_clf = load(rocket_instances_path.joinpath('rocket_1337_clf.pkl'))

    # Batch is entire dataset
    train_batch = next(iter(dm.train_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))

    # Perform ROCKET transformation stage on train, val, and test data
    train_features = rocket(train_batch.x)
    test_features = rocket(test_batch.x)
    val_features = rocket(val_batch.x)

    # Make predictions on validation set
    val_pred = ridge_clf.predict(val_features)

    # The RidgeClassifier maps the targets to {-1, 1}, but our labels are {0, 1}
    val_pred[val_pred < 0] = 0

    # val_pred is now a np array with float predictions in {0, 1}, might be useful to have as tensor - like the features
    val_pred = tensor(val_pred)

    # Targets may be accessed like this
    val_labels = val_batch.y


if __name__ == '__main__':
    # 1337 is the seed used for the best performing rocket model.
    main(1337)
