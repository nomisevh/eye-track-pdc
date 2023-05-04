from matplotlib import pyplot as plt
from numpy import arange
from sklearn.linear_model import RidgeClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
from torch import tensor
from torchmetrics.functional.classification import multiclass_f1_score, binary_accuracy
from yaml import FullLoader, load as load_yaml

from datamodule import KIDataModule
from model_selection import get_attribute_power
from models.rocket import ROCKET
from utils.const import SEED
from utils.ki import SACCADE, AXIS
from utils.metric import vote_aggregation, max_f1_score
from utils.misc import set_random_state
from utils.path import config_path
from utils.visualize import visualize_latent_space


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
    ridge_clf = RidgeClassifier(alpha=1e4, random_state=SEED)

    # Initialize dimensionality reducer
    tsne = TSNE(n_components=2, perplexity=50)

    # Batch is entire dataset
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))

    # Perform ROCKET transformation stage on train and test data
    train_features = rocket(train_batch.x)
    test_features = rocket(test_batch.x)
    val_features = rocket(val_batch.x)

    # Fit classifier to the rocket features
    ridge_clf.fit(train_features, train_batch.y)

    # Make predictions on test set
    test_pred = ridge_clf.predict(test_features)
    val_pred = ridge_clf.predict(val_features)

    # The RidgeClassifier maps the targets to {-1, 1}, but our labels are {0, 1}
    test_pred[test_pred < 0] = 0

    test_trial_probs = tensor(test_pred)
    val_trial_probs = tensor(val_pred)

    # Aggregate predictions to the subject level
    _, test_subject_labels, test_subject_probs = vote_aggregation(segment_scores=test_trial_probs, labels=test_batch.y,
                                                                  aggregate_by=test_batch.z)
    _, val_subject_labels, val_subject_probs = vote_aggregation(segment_scores=val_trial_probs, labels=val_batch.y,
                                                                aggregate_by=val_batch.z)

    # Find the best subject threshold with respect to the unweighted f1 score on the validation set
    _, val_subject_threshold = max_f1_score(val_subject_probs, val_subject_labels)
    # Find the best subject threshold with respect to the unweighted f1 score on the test set
    _, test_subject_threshold = max_f1_score(test_subject_probs, test_subject_labels)

    # Compute unweighted F1 and accuracy for trials and subjects on the test set
    test_trial_uf1_score = multiclass_f1_score((test_trial_probs >= 0.5).long(), test_batch.y.long(), num_classes=2,
                                               average='macro')
    test_subject_uf1_score_best = multiclass_f1_score((test_subject_probs >= test_subject_threshold).long(),  # noqa
                                                      test_subject_labels.long(), num_classes=2, average='macro')
    test_subject_uf1_score_val = multiclass_f1_score((test_subject_probs >= val_subject_threshold).long(),  # noqa
                                                     test_subject_labels.long(), num_classes=2, average='macro')
    val_subject_uf1_score = multiclass_f1_score((val_subject_probs >= val_subject_threshold).long(),  # noqa
                                                val_subject_labels.long(), num_classes=2, average='macro')

    # Report the results on trial level
    print(f"test trial uF1: {test_trial_uf1_score :.4f}")
    print(f"test trial accuracy: {binary_accuracy(test_trial_probs, test_batch.y):.2%}"
          f" with threshold 0.5")

    # Report the results on subject level with the best threshold found on the validation set
    print(f"test subject uF1 (val threshold): {test_subject_uf1_score_val:.4f}")
    print(f"test subject accuracy (val threshold): "
          f"{binary_accuracy(test_subject_probs, test_subject_labels, threshold=val_subject_threshold):.2%}"
          f" with threshold {val_subject_threshold:.2f}")

    # Report the results on subject level with the best threshold found on the test set
    print(f"test subject uF1 (test threshold): {test_subject_uf1_score_best:.4f}")
    print(f"test subject accuracy (test threshold): "
          f"{binary_accuracy(test_subject_probs, test_subject_labels, threshold=test_subject_threshold):.2%}"
          f" with threshold {test_subject_threshold:.2f}")

    print(f"val subject uF1: {val_subject_uf1_score:.4f} with threshold {val_subject_threshold:.2f}")
    print(f"val subject accuracy: "
          f"{binary_accuracy(val_subject_probs, val_subject_labels, threshold=val_subject_threshold):.2%}")

    # Perform attribute-based subgroup evaluation
    attribute_power = get_attribute_power(test_batch, test_trial_probs, threshold=0.5)
    attribute_power = {k: f'{v:.2%}' for k, v in attribute_power.items()}
    print(f'Attribute power: {attribute_power}')

    figure = ConfusionMatrixDisplay.from_predictions(test_batch.y, (test_trial_probs >= 0.5).int(),
                                                     display_labels=dm.class_names(),
                                                     cmap='Blues')
    figure.ax_.set_title('Trial Classification with ROCKET')
    figure.figure_.set_size_inches(4, 3)
    plt.tight_layout()
    figure.figure_.savefig('trial_classification_rocket.svg', format='svg', dpi=1200)
    plt.show()

    figure = ConfusionMatrixDisplay.from_predictions(test_subject_labels,
                                                     (test_subject_probs >= test_subject_threshold).int(),  # noqa
                                                     display_labels=dm.class_names(),
                                                     cmap='Blues')

    figure.ax_.set_title('Subject Classification with ROCKET')
    figure.figure_.set_size_inches(4, 3)
    plt.tight_layout()
    figure.figure_.savefig('subject_classification_rocket.svg', format='svg', dpi=1200)
    plt.show()

    return
    # Visualize the latent neighborhoods with TSNE
    plot_latent_neighborhood(rocket, test_batch, dm)


def plot_latent_neighborhood(model, eval_batch, dm):
    tsne = TSNE(n_components=2, perplexity=50)
    eval_embeddings = model(eval_batch.x)
    manifold = tsne.fit_transform(eval_embeddings.detach())
    visualize_latent_space(manifold, eval_batch.s, SACCADE.keys(), show=True,
                           title='Latent Neighborhood of Unseen Trials (Saccade)')
    visualize_latent_space(manifold, eval_batch.a, AXIS.keys(), show=True,
                           title='Latent Neighborhood of Unseen Trials (Axis)')
    visualize_latent_space(manifold, eval_batch.y, dm.class_names(), show=True,
                           title='Latent Neighborhood of Unseen Trials')
    visualize_latent_space(manifold, eval_batch.g, dm.group_names(), show=True,
                           title='Latent Neighborhood of Unseen Trials')

    # Use modulo to get the index of each segment in the flattened batch, relative to every experiment
    saccade_time = 1
    inter_saccade_time = (440 / 300) + 1.2
    index_in_experiment = (arange(eval_embeddings.shape[0]) % 11) * (
            inter_saccade_time + saccade_time) + inter_saccade_time + saccade_time + 1
    fig, ax = plt.subplots(figsize=(7, 5), dpi=400)
    scatter = ax.scatter(manifold[:, 0], manifold[:, 1],
                         c=index_in_experiment,
                         cmap='plasma', s=20)
    # Add colorbar with unit seconds
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Start Time of Trial Within Session (s)')
    ax.set_title('Latent Neighborhood of Unseen Trials')
    plt.show()


if __name__ == '__main__':
    main(1337)

    # for seed in [42, 1337, 9000, 1, 2]:
    #     main(seed)
