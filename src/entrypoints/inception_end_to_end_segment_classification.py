from warnings import filterwarnings

import matplotlib.pyplot as plt
import torch
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics.functional.classification import binary_accuracy
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
from model_selection import get_attribute_power
from models.inceptiontime import EndToEndInceptionTimeClassifier
from utils.const import SEED
from utils.ki import SACCADE, AXIS
from utils.metric import vote_aggregation, unweighted_binary_average_precision, max_f1_score
from utils.misc import set_random_state
from utils.path import config_path, log_path, checkpoint_path
from utils.visualize import visualize_latent_space

TAGS = ['Final Results', 'Triplet/IPS Ablation']


def main():
    filterwarnings("ignore", category=PossibleUserWarning)
    filterwarnings("ignore", category=FutureWarning)
    set_random_state(SEED)

    # Load configs
    with open(config_path.joinpath('leif.yaml')) as reader:
        processor_config = load_yaml(reader, Loader=FullLoader)
    with open(config_path.joinpath('inception.yaml')) as reader:
        inception_config = load_yaml(reader, Loader=FullLoader)
    with open(config_path.joinpath('neptune.yaml')) as reader:
        neptune_config = load_yaml(reader, Loader=FullLoader)

    dm = KIDataModule(processor_config=processor_config,
                      use_triplets=True,
                      exclude=['vert'],
                      binary_classification=True,
                      batch_size=256,
                      val_size=0.2,
                      ips=True)
    dm.setup('fit')
    dm.setup('test')

    model = EndToEndInceptionTimeClassifier(num_classes=1, triplet_loss=True, seed=SEED, **inception_config)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor='val_uap', every_n_epochs=1, mode='max')
    trainer = Trainer(accelerator='auto',
                      max_epochs=300,
                      default_root_dir=log_path,
                      log_every_n_steps=1,
                      logger=NeptuneLogger(log_model_checkpoints=False, **neptune_config, tags=TAGS),
                      callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=dm)

    model = EndToEndInceptionTimeClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)

    # model = EndToEndInceptionTimeClassifier.load_from_checkpoint(checkpoint_path.joinpath('epoch=200-step=804.ckpt'))

    model.freeze()

    dm.set_use_triplets(False)
    dm.batch_size = -1
    val_batch = next(iter(dm.val_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))

    # Make predictions on the validation and test set
    val_trial_probs = torch.concatenate(trainer.predict(model, dataloaders=dm.val_dataloader()))
    test_trial_probs = torch.concatenate(trainer.predict(model, datamodule=dm))
    # Aggregate predictions to the subject level
    _, test_subject_labels, test_subject_probs = vote_aggregation(segment_scores=test_trial_probs, labels=test_batch.y,
                                                                  aggregate_by=test_batch.z)
    _, val_subject_labels, val_subject_probs = vote_aggregation(segment_scores=val_trial_probs, labels=val_batch.y,
                                                                aggregate_by=val_batch.z)

    # Find the best trial threshold with respect to the unweighted f1 score on the validation set
    _, val_trial_threshold = max_f1_score(val_trial_probs, val_batch.y)
    # Find the best subject threshold with respect to the unweighted f1 score on the validation set
    _, val_subject_threshold = max_f1_score(val_subject_probs, val_subject_labels)

    # For fun do the same for the test set
    _, test_trial_threshold = max_f1_score(test_trial_probs, test_batch.y)
    _, test_subject_threshold = max_f1_score(test_subject_probs, test_subject_labels)

    # Compute unweighted average precision and accuracy for trials and subjects on the test set
    print(f"test trial uAP: {unweighted_binary_average_precision(test_trial_probs, test_batch.y):.4f}")
    print(f"test trial accuracy: {binary_accuracy(test_trial_probs, test_batch.y, threshold=val_trial_threshold):.2%}"
          f" with threshold {val_trial_threshold:.2f}")

    print(f"test subject uAP: {unweighted_binary_average_precision(test_subject_probs, test_subject_labels)}:.4f")
    print(f"test subject accuracy: "
          f"{binary_accuracy(test_subject_probs, test_subject_labels, threshold=val_subject_threshold):.2%}"
          f" with threshold {val_subject_threshold:.2f}")

    # Perform attribute-based subgroup evaluation
    attribute_power = get_attribute_power(test_batch, test_trial_probs, val_trial_threshold)
    attribute_power = {k: f'{v:.2%}' for k, v in attribute_power.items()}
    print(f'Attribute power: {attribute_power}')

    figure = ConfusionMatrixDisplay.from_predictions(test_batch.y, (test_trial_probs >= val_trial_threshold).int(),
                                                     display_labels=dm.class_names(),
                                                     cmap='Blues')
    figure.ax_.set_title('Trial Classification')
    plt.show()

    figure = ConfusionMatrixDisplay.from_predictions(test_subject_labels,
                                                     (test_subject_probs >= val_subject_threshold).int(),  # noqa
                                                     display_labels=dm.class_names(),
                                                     cmap='Blues')
    figure.ax_.set_title('Subject Classification')
    plt.show()

    # Visualize the latent neighborhoods with TSNE
    plot_latent_neighborhood(model, test_batch, dm)


def plot_latent_neighborhood(model, eval_batch, dm):
    tsne = TSNE(n_components=2, perplexity=30)
    eval_embeddings, _ = model(eval_batch.x)
    manifold = tsne.fit_transform(eval_embeddings.detach())
    visualize_latent_space(manifold, eval_batch.s, SACCADE.keys(), show=True,
                           title='Latent Neighborhood of Test Segments (Saccade)')
    visualize_latent_space(manifold, eval_batch.a, AXIS.keys(), show=True,
                           title='Latent Neighborhood of Test Segments (Axis)')
    visualize_latent_space(manifold, eval_batch.y, dm.class_names(), show=True,
                           title='Latent Neighborhood of Segments')
    visualize_latent_space(manifold, eval_batch.g, dm.group_names(), show=True,
                           title='Latent Neighborhood of Segments')


if __name__ == '__main__':
    main()
