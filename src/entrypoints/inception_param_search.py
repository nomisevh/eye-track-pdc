from itertools import chain
from warnings import filterwarnings

from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from dataset import KIDataset, train_test_split_stratified
from model_selection import grid_search_2d, Validator
from models.inceptiontime import LitInceptionTimeModel
from processor.processor import Leif
from utils.const import SEED
from utils.misc import set_random_state
from utils.path import config_path, log_path, checkpoint_path


def main():
    filterwarnings("ignore", category=PossibleUserWarning)
    set_random_state(SEED)

    # Load configs
    with open(config_path.joinpath('leif.yaml')) as reader:
        processor_config = load_yaml(reader, Loader=FullLoader)
    with open(config_path.joinpath('inception.yaml')) as reader:
        inception_config = load_yaml(reader, Loader=FullLoader)

    # Callback for searching the parameters of inceptionTime
    class Callback(Validator.Callback):
        def __call__(self, train_ds, val_ds, iteration, **kwargs):
            # Construct data module
            dm = KIDataModule(
                train_ds=train_ds,
                val_ds=val_ds,
                use_triplets=True,
                binary_classification=True,
                batch_size=256
            )
            dm.setup(stage='fit')

            checkpoint_filename = f'{"-".join(map(str, chain(*kwargs.items())))}-iter-{iteration}'

            # Overwrite InceptionTime config
            train_inception_time(dm,
                                 inception_config={**inception_config, **kwargs},
                                 checkpoint_filename=checkpoint_filename)

            metric = fit_and_predict_clf(dm, f'{checkpoint_filename}.ckpt')

            return metric

    # Prepare data
    train_val_ds = KIDataset(data_processor=Leif(processor_config), train=True, use_triplets=True)

    # Initialize validator
    validator = Validator(num_random_inits=2, num_folds=5, splitter=train_test_split_stratified)

    # Perform grid search
    _ = grid_search_2d(validator, Callback(), train_val_ds,
                       bottleneck_channels=[1, 2, 0],
                       num_pred_classes=[64, 128, 256])


def train_inception_time(dm, inception_config, checkpoint_filename):
    inception_time = LitInceptionTimeModel(**inception_config)
    trainer = Trainer(accelerator='auto',
                      max_epochs=100,
                      default_root_dir=log_path,
                      log_every_n_steps=1,
                      callbacks=[ModelCheckpoint(dirpath=checkpoint_path,
                                                 monitor='val_loss',
                                                 every_n_epochs=1,
                                                 filename=checkpoint_filename)],
                      )
    trainer.fit(inception_time, datamodule=dm)


def fit_and_predict_clf(dm: KIDataModule, checkpoint_filename):
    inception_time = LitInceptionTimeModel.load_from_checkpoint(checkpoint_path.joinpath(checkpoint_filename))

    # Freeze parameters of the encoder
    inception_time.freeze()

    # Initialize classifiers
    forest_clf = RandomForestClassifier(
        n_jobs=1,
        n_estimators=200,
        max_samples=0.826,
        max_depth=2
    )

    # Configure data module for fitting the entire dataset with classifier
    dm.set_use_triplets(False)
    dm.batch_size = -1
    # Batch is entire dataset
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))

    # Fit
    train_embeddings = inception_time(train_batch.x)
    forest_clf.fit(train_embeddings.detach(), train_batch.y)

    # Evaluate
    val_embeddings = inception_time(val_batch.x)
    pred = forest_clf.predict(val_embeddings.detach())

    # Compute and return metric
    return f1_score(val_batch.y, pred, average='macro')


if __name__ == '__main__':
    main()
