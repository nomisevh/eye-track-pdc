from warnings import filterwarnings

from lightning_fabric.utilities.warnings import PossibleUserWarning
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from dataset import KIDataset, train_test_split_stratified
from model_selection import grid_search_2d, Validator
from models.rocket import ROCKET
from processor.processor import Leif
from utils.const import SEED
from utils.misc import set_random_state
from utils.path import config_path


def main():
    filterwarnings("ignore", category=PossibleUserWarning)
    set_random_state(SEED)

    # Load configs
    with open(config_path.joinpath('leif.yaml')) as reader:
        processor_config = load_yaml(reader, Loader=FullLoader)

    # Callback for searching the parameters of inceptionTime
    class Callback(Validator.Callback):
        def __call__(self, train_ds, val_ds, iteration, **kwargs):
            # Construct data module
            dm = KIDataModule(
                train_ds=train_ds,
                val_ds=val_ds,
                use_triplets=False,
                binary_classification=True,
                batch_size=-1,
            )
            dm.setup(stage='fit')

            # Overwrite InceptionTime config
            rocket = ROCKET(c_in=dm.train_ds.x.shape[1], seq_len=dm.train_ds.x.shape[2], n_kernels=kwargs['n_kernels'],
                            normalize=True)
            clf = RidgeClassifier(alpha=kwargs['alpha'], random_state=SEED)

            # Train and test batches are entire dataset
            train_batch = next(iter(dm.train_dataloader()))
            eval_batch = next(iter(dm.val_dataloader()))

            # Perform ROCKET transformation stage on train and eval data
            train_features = rocket(train_batch.x)
            # Set to eval mode, as to use statistics from train batch for normalization
            rocket.train = False
            eval_features = rocket(eval_batch.x)

            clf.fit(train_features, train_batch.y)

            # Make predictions on eval set
            pred = clf.predict(eval_features)

            # The RidgeClassifier maps the targets to {-1, 1}, but our labels are {0, 1}
            pred[pred < 0] = 0

            f1 = f1_score(eval_batch.y, pred, average='macro')
            return f1

    # Prepare data
    train_val_ds = KIDataset(data_processor=Leif(processor_config), exclude=['vert'], train=True, use_triplets=False)

    # Initialize validator
    validator = Validator(num_random_inits=1, num_folds=5, splitter=train_test_split_stratified)

    # Perform grid search
    _ = grid_search_2d(validator, Callback(), train_val_ds,
                       n_kernels=[100, 1000, 10000],
                       alpha=[1e3, 1e4, 1e5]),


if __name__ == '__main__':
    main()
