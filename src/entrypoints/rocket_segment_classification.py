from typing import Sequence

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model._base import LinearModel
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch import Tensor
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from dataset import Signature
from models.rocket import ROCKET
from utils.const import SEED
from utils.misc import set_random_state
from utils.path import config_path
from utils.visualize import plot_top_eigenvalues, visualize_latent_space


def main():
    set_random_state(SEED)

    # Load configs
    with open(config_path.joinpath('leif.yaml')) as reader:
        processor_config = load_yaml(reader, Loader=FullLoader)
    with open(config_path.joinpath('rocket.yaml')) as reader:
        rocket_config = load_yaml(reader, Loader=FullLoader)

    dm = KIDataModule(processor_config=processor_config,
                      bundle_as_experiments=False,
                      binary_classification=True,
                      batch_size=-1)
    dm.setup('fit')
    dm.setup('test')

    # Initialize Rocket
    rocket = ROCKET(c_in=dm.train_ds.x.shape[1],
                    seq_len=dm.train_ds.x.shape[2],
                    **rocket_config)

    # Initialize Classifiers
    ridge_clf = RidgeClassifier(alpha=1e5, random_state=SEED)
    forest_clf = RandomForestClassifier(
        random_state=SEED,
        n_jobs=1,
        n_estimators=200,
        max_samples=0.5,
        max_depth=2)

    # Initialize dimensionality reducer
    tsne = TSNE(n_components=2, perplexity=50)

    # Batch is entire dataset
    train_batch = next(iter(dm.train_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))

    # Perform ROCKET transformation stage on train and test data
    train_features = rocket(train_batch.x)
    test_features = rocket(test_batch.x)

    # Visualize how many principal components are required to represent data
    plot_top_eigenvalues(test_features)

    # Visualize the latent neighborhoods with TSNE
    manifold = tsne.fit_transform(test_features)
    visualize_latent_space(manifold, test_batch, dm.class_names())

    for clf_name, clf in [('ridge classifier', ridge_clf), ('random forest classifier', forest_clf)]:
        # Fit classifier to the rocket features
        clf.fit(train_features, train_batch.y)

        # Evaluate on the test set
        evaluate(test_features, clf, test_batch, dm.class_names(), clf_name)


def evaluate(test_features: Tensor, clf: LinearModel, test_batch: Signature, labels: Sequence[str], clf_name: str):
    # Make predictions on test set
    pred = clf.predict(test_features)

    # The RidgeClassifier maps the targets to {-1, 1}, but our labels are {0, 1}
    pred[pred < 0] = 0

    # Compute metrics
    report = classification_report(test_batch.y, pred, target_names=labels)

    # Construct the confusion matrix
    cf_matrix = confusion_matrix(test_batch.y, pred)

    # Plot metrics and display confusion matrix
    print(f'results for {clf_name}')
    print(f"mean F1 score: {f1_score(test_batch.y, pred, average='macro')}")
    print(f'segment classification on test set:\n{report}')
    figure = ConfusionMatrixDisplay(cf_matrix, display_labels=labels).plot(cmap='Blues')
    figure.ax_.set_title('Segment Classification on Test Set')
    plt.show()


if __name__ == '__main__':
    main()
