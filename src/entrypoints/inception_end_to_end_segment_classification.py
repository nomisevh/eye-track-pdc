import matplotlib.pyplot as plt
from numpy import arange, zeros, argmax, concatenate
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
from model_selection import get_attribute_power
from models.inceptiontime import EndToEndInceptionTimeClassifier
from utils.const import SEED
from utils.ki import SACCADE, AXIS
from utils.metric import patient_soft_accuracy
from utils.misc import set_random_state
from utils.path import config_path, log_path, checkpoint_path
from utils.visualize import visualize_latent_space


def main():
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

    model = EndToEndInceptionTimeClassifier(num_classes=1, triplet_loss=True, **inception_config)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor='val_f1', every_n_epochs=1, mode='max')
    trainer = Trainer(accelerator='auto',
                      max_epochs=500,
                      default_root_dir=log_path,
                      log_every_n_steps=1,
                      logger=NeptuneLogger(log_model_checkpoints=False, **neptune_config),
                      callbacks=[checkpoint_callback])

    # trainer.fit(model, datamodule=dm)
    dm.set_use_triplets(False)

    # model = EndToEndInceptionTimeClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)
    model = EndToEndInceptionTimeClassifier.load_from_checkpoint(checkpoint_path.joinpath('epoch=200-step=804.ckpt'))

    # Freeze parameters of the encoder
    model.freeze()

    dm.batch_size = -1
    val_batch = next(iter(dm.test_dataloader()))

    # Visualize the latent neighborhoods with TSNE
    tsne = TSNE(n_components=2, perplexity=30)
    eval_embeddings, _ = model(val_batch.x)
    manifold = tsne.fit_transform(eval_embeddings.detach())
    visualize_latent_space(manifold, val_batch.s, SACCADE.keys(), show=True,
                           title='Latent Neighborhood of Test Segments (Saccade)')
    visualize_latent_space(manifold, val_batch.a, AXIS.keys(), show=True,
                           title='Latent Neighborhood of Test Segments (Axis)')
    visualize_latent_space(manifold, val_batch.y, dm.class_names(), show=True,
                           title='Latent Neighborhood of Test Segments (Class)')
    visualize_latent_space(manifold, val_batch.g, dm.group_names(), show=True,
                           title='Latent Neighborhood of Test Segments (Group)')

    pred = concatenate(trainer.predict(model, datamodule=dm))

    # Compute and return metric
    print(f"f1: {f1_score(val_batch.y, pred, average='macro')}")
    print(f"accuracy: {accuracy_score(val_batch.y, pred)}")

    figure = ConfusionMatrixDisplay.from_predictions(val_batch.y, pred, display_labels=dm.class_names(), cmap='Blues')
    figure.ax_.set_title('Segment Classification')
    plt.show()

    # Compute attribute importance
    attribute_power = get_attribute_power(val_batch, pred)
    print(attribute_power)

    thresholds = arange(0, 1, 0.01)
    f1_scores = zeros(len(thresholds))
    accuracy_scores = zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        patient_pred, patient_label, patient_acc = patient_soft_accuracy(segment_scores=pred, y=val_batch.y,
                                                                         z=val_batch.z, threshold=threshold)
        f1_scores[i] = (f1_score(patient_label, patient_pred, average='macro'))
        accuracy_scores[i] = accuracy_score(patient_label, patient_pred)

    patient_pred, patient_label, patient_acc = patient_soft_accuracy(segment_scores=pred, y=val_batch.y,
                                                                     z=val_batch.z,
                                                                     threshold=thresholds[argmax(f1_scores)])

    plt.plot(thresholds, f1_scores)
    plt.show()
    print(f"max f1: {max(f1_scores)} with threshold: {thresholds[argmax(f1_scores)]}")
    print(f'patient accuracy: {accuracy_scores[argmax(f1_scores)]}')
    figure = ConfusionMatrixDisplay.from_predictions(patient_label, patient_pred, display_labels=dm.class_names(),
                                                     cmap='Blues')
    figure.ax_.set_title('Aggregate Classification')
    plt.show()


def compute_attribute_importance(attribute, pred, labels):
    return f1_score(labels, pred, average='macro', sample_weight=attribute) - \
           f1_score(labels, pred, average='macro')


if __name__ == '__main__':
    main()
