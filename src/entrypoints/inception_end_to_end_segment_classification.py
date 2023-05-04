from warnings import filterwarnings

import matplotlib.pyplot as plt
import torch
from lightning_fabric.utilities.warnings import PossibleUserWarning
from numpy import arange
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics.functional.classification import binary_accuracy, multiclass_f1_score
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
from model_selection import get_attribute_power
from models.inceptiontime import EndToEndInceptionTimeClassifier
from utils.const import SEED
from utils.ki import SACCADE
from utils.metric import vote_aggregation, max_f1_score
from utils.misc import set_random_state
from utils.path import config_path, log_path, checkpoint_path
from utils.visualize import visualize_latent_space, separate_latent_space_by_attr

TAGS = ['Final Results', 'Main Results v3']


def main(seed):
    filterwarnings("ignore", category=PossibleUserWarning)
    filterwarnings("ignore", category=FutureWarning)
    set_random_state(seed)

    # Load configs
    with open(config_path.joinpath('leif.yaml')) as reader:
        processor_config = load_yaml(reader, Loader=FullLoader)
    with open(config_path.joinpath('inception.yaml')) as reader:
        inception_config = load_yaml(reader, Loader=FullLoader)
    with open(config_path.joinpath('neptune.yaml')) as reader:
        neptune_config = load_yaml(reader, Loader=FullLoader)

    dm = KIDataModule(processor_config=processor_config,
                      use_triplets=False,
                      exclude=['vert'],
                      binary_classification=True,
                      batch_size=256,
                      val_size=0.2,
                      ips=False)
    dm.setup('fit')
    dm.setup('test')

    model = EndToEndInceptionTimeClassifier(num_classes=1, seed=seed, **inception_config)

    logger = NeptuneLogger(log_model_checkpoints=False, **neptune_config, tags=TAGS)
    print('waiting for neptune to initialize...')
    logger.experiment.wait()
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor='val_f1', every_n_epochs=1, mode='max',
                                          filename=logger.version + '-{epoch}')
    trainer = Trainer(accelerator='auto',
                      max_epochs=300,
                      default_root_dir=log_path,
                      log_every_n_steps=1,
                      logger=logger,
                      callbacks=[checkpoint_callback])
    # trainer.fit(model, datamodule=dm)

    # model = EndToEndInceptionTimeClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)

    model = EndToEndInceptionTimeClassifier.load_from_checkpoint(checkpoint_path.joinpath('PDC-378-epoch=18.ckpt'))

    model.freeze()

    dm.set_use_triplets(False)
    dm.batch_size = -1
    val_batch = next(iter(dm.val_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))

    # Make predictions on the validation and test set
    val_trial_probs = torch.concatenate(trainer.predict(model, dataloaders=dm.val_dataloader()))
    test_trial_probs = torch.concatenate(trainer.predict(model, datamodule=dm))

    # OBS temporary only use prosaccade data for evaluation
    # val_trial_probs = val_trial_probs[val_batch.s == 0]
    # val_batch = Signature(val_batch.x[val_batch.s == 0], val_batch.y[val_batch.s == 0], val_batch.z[val_batch.s == 0],
    #                       val_batch.r[val_batch.s == 0], val_batch.a[val_batch.s == 0], val_batch.s[val_batch.s == 0],
    #                       val_batch.g[val_batch.s == 0])
    # test_trial_probs = test_trial_probs[test_batch.s == 0]
    # test_batch = Signature(test_batch.x[test_batch.s == 0], test_batch.y[test_batch.s == 0],
    #                        test_batch.z[test_batch.s == 0], test_batch.r[test_batch.s == 0],
    #                        test_batch.a[test_batch.s == 0], test_batch.s[test_batch.s == 0],
    #                        test_batch.g[test_batch.s == 0])

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
    attribute_power = get_attribute_power(test_batch, test_trial_probs)
    attribute_power = {k: f'{v:.2%}' for k, v in attribute_power.items()}
    print(f'Attribute power: {attribute_power}')

    figure = ConfusionMatrixDisplay.from_predictions(test_batch.y, (test_trial_probs >= 0.5).int(),
                                                     display_labels=dm.class_names(),
                                                     cmap='Blues')
    figure.ax_.set_title('Trial Classification with InceptionTime')
    figure.figure_.set_size_inches(4, 3)
    plt.tight_layout()
    # figure.figure_.savefig('trial_classification_inceptiontime.svg', format='svg', dpi=1200)
    plt.show()

    figure = ConfusionMatrixDisplay.from_predictions(test_subject_labels,
                                                     (test_subject_probs >= test_subject_threshold).int(),  # noqa
                                                     display_labels=dm.class_names(),
                                                     cmap='Blues')

    figure.ax_.set_title('Subject Classification with InceptionTime')
    # Set the figure size
    figure.figure_.set_size_inches(4, 3)
    plt.tight_layout()
    # figure.figure_.savefig('subject_classification_inceptiontime.svg', format='svg', dpi=1200)
    plt.show()

    # Visualize the latent neighborhoods with TSNE
    plot_latent_neighborhood(model, test_batch, dm)


def plot_latent_neighborhood(model, eval_batch, dm):
    tsne = TSNE(n_components=2, perplexity=70)
    eval_embeddings, _ = model(eval_batch.x)
    manifold = tsne.fit_transform(eval_embeddings.detach())
    fig, _ = separate_latent_space_by_attr(manifold, eval_batch.y, eval_batch.s, dm.class_names(), SACCADE.keys(),
                                           title='Latent Representation of Test Set')

    fig, ax = visualize_latent_space(manifold, eval_batch.y, dm.class_names(), show=False,
                                     title='Latent Representation of Test Set')
    plt.tight_layout()
    fig.savefig('latent_representation_class.svg', format='svg', dpi=1200)
    plt.show()

    return

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
    main(SEED)

    # for seed in [42, 1337, 9000, 1, 2]:
    #    main(seed)
