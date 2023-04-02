from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from torch import nn
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from dataset import KIDataset, train_test_split_stratified
from model_selection import grid_search_2d, Validator
from models.inceptiontime import LitInceptionTime
from processor.processor import Leif
from utils.const import SEED
from utils.misc import set_random_state
from utils.path import config_path, checkpoint_path


def main():
    set_random_state(SEED)

    # Load configs
    with open(config_path.joinpath('leif.yaml')) as reader:
        processor_config = load_yaml(reader, Loader=FullLoader)

    # Load InceptionTime checkpoint
    model = LitInceptionTime.load_from_checkpoint(checkpoint_path.joinpath('winner.ckpt'))
    # Freeze parameters of the encoder
    model.freeze()

    # Callback for searching the parameters of inceptionTime
    class Callback(Validator.Callback):
        def __call__(self, train_ds, val_ds, iteration, **kwargs):
            # Construct data module
            dm = KIDataModule(
                train_ds=train_ds,
                val_ds=val_ds,
                use_triplets=False,
                binary_classification=True,
                batch_size=-1
            )
            dm.setup(stage='fit')

            metric = fit_and_predict_clf(model, dm, **kwargs)

            return metric

    # Prepare data
    train_val_ds = KIDataset(data_processor=Leif(processor_config), train=True, use_triplets=True)

    # Initialize validator
    validator = Validator(num_random_inits=1, num_folds=5, splitter=train_test_split_stratified)

    # Perform grid search
    _ = grid_search_2d(validator, Callback(), train_val_ds,
                       max_samples=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       n_estimators=[100, 200, 300, 400])


def fit_and_predict_clf(feature_extractor: nn.Module, dm: KIDataModule, **kwargs):
    params = {
        'n_jobs': 1,
        'n_estimators': 200,
        'max_samples': 0.6,
        'max_depth': 2,
        **kwargs
    }

    # Initialize classifiers
    clf = RandomForestClassifier(**params)

    # Batch is entire dataset
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))

    # Fit
    train_embeddings = feature_extractor(train_batch.x)
    clf.fit(train_embeddings.detach(), train_batch.y)

    # Evaluate
    val_embeddings = feature_extractor(val_batch.x)
    pred = clf.predict(val_embeddings.detach())

    # Compute and return metric
    return f1_score(val_batch.y, pred, average='macro')


if __name__ == '__main__':
    main()
