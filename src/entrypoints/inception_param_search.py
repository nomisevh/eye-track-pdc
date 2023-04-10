from warnings import filterwarnings

from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning import Trainer
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from dataset import KIDataset, train_test_split_stratified
from model_selection import grid_search_2d, Validator
from models.inceptiontime import EndToEndInceptionTimeClassifier
from processor.processor import Leif
from utils.const import SEED
from utils.metric import ValidationMetricCallback
from utils.misc import set_random_state
from utils.path import config_path, log_path


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
                use_triplets=False,
                binary_classification=True,
                batch_size=256,
            )

            # Overwrite InceptionTime config
            model = EndToEndInceptionTimeClassifier(**{**inception_config, **kwargs})
            validation_metric_callback = ValidationMetricCallback(metric='val_uap', mode='max')
            trainer = Trainer(accelerator='auto',
                              max_epochs=100,
                              default_root_dir=log_path,
                              log_every_n_steps=1,
                              callbacks=[validation_metric_callback],
                              )
            dm.setup(stage='fit')
            trainer.fit(model, datamodule=dm)

            return validation_metric_callback.best.item()

    # Prepare data
    train_val_ds = KIDataset(data_processor=Leif(processor_config), train=True, use_triplets=False)

    # Initialize validator
    validator = Validator(num_random_inits=2, num_folds=5, splitter=train_test_split_stratified)

    # Perform grid search
    _ = grid_search_2d(validator, Callback(), train_val_ds,
                       depth=[2],
                       bottleneck_dim=[46])


if __name__ == '__main__':
    main()
