from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

from utils.data import binarize


def train_eval_rocket_segment(train_ds, eval_ds, rocket, clf, binary=True):
    # Binarize dataset after split to make sure split is stratified w.r.t all three classes
    if binary:
        for ds in [train_ds, eval_ds]:
            binarize(ds)

    # Initialize Dataloaders
    train_dl = DataLoader(train_ds,
                          batch_size=train_ds.x.shape[0],
                          sampler=ImbalancedDatasetSampler(train_ds, callback_get_label=lambda item: item.y))
    val_dl = DataLoader(eval_ds, batch_size=eval_ds.x.shape[0])

    # Batch is entire dataset
    train_batch = next(iter(train_dl))
    eval_batch = next(iter(val_dl))

    # Perform ROCKET transformation stage on train set
    features = rocket(train_batch.x)

    # Fit Classifier
    clf.fit(features, train_batch.y.numpy())

    # Perform ROCKET transformation stage on validation set
    test_features = rocket(eval_batch.x)

    # Make predictions on validation set
    pred = clf.predict(test_features)

    # The RidgeClassifier maps the targets to {-1, 1}, but our labels are {0, 1}
    pred[pred < 0] = 0

    return pred, eval_batch
