from sklearn.linear_model import RidgeClassifier
from torch import tensor
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from models.rocket import ROCKET
from utils.const import SEED
from utils.eval import evaluate
from utils.misc import set_random_state
from utils.path import config_path


def main(seed):
    set_random_state(seed)

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
    rocket = ROCKET(c_in=dm.train_ds.x.shape[1],
                    seq_len=dm.train_ds.x.shape[2],
                    **rocket_config)

    # Initialize Classifiers
    ridge_clf = RidgeClassifier(alpha=1e3, random_state=SEED)

    # Batch is entire dataset
    train_batch = next(iter(dm.train_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))

    # Perform ROCKET transformation stage on train and test data
    train_features = rocket(train_batch.x)
    test_features = rocket(test_batch.x)

    # Fit classifier to the rocket features
    ridge_clf.fit(train_features, train_batch.y)

    # Make predictions on test set
    test_pred = ridge_clf.predict(test_features)

    # The RidgeClassifier maps the targets to {-1, 1}, but our labels are {0, 1}
    test_pred[test_pred < 0] = 0

    test_trial_probs = tensor(test_pred)

    evaluate(test_batch, test_trial_probs, test_features, dm.class_names(), model_name='ROCKET')


if __name__ == '__main__':
    main(1337)

    # for seed in [42, 1337, 9000, 1, 2]:
    #     main(seed)
