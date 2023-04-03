import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
from models.inceptiontime import LitInceptionTime
from utils.const import SEED
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
                      binary_classification=True,
                      batch_size=256)
    dm.setup('fit')

    model = LitInceptionTime(**inception_config)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor='val_loss', every_n_epochs=1)
    trainer = Trainer(accelerator='auto',
                      max_epochs=100,
                      default_root_dir=log_path,
                      log_every_n_steps=1,
                      logger=NeptuneLogger(log_model_checkpoints=False, **neptune_config),
                      callbacks=[checkpoint_callback])

    trainer.fit(model, datamodule=dm)

    # model = LitInceptionTime.load_from_checkpoint(checkpoint_path.joinpath('use_this.ckpt'))
    model = LitInceptionTime.load_from_checkpoint(checkpoint_callback.best_model_path)

    dm.set_use_triplets(False)
    dm.batch_size = -1

    # trainer.fit(classifier, datamodule=dm)

    # Freeze parameters of the encoder
    model.freeze()

    # Initialize classifiers
    forest_clf = RandomForestClassifier(
        n_jobs=1,
        n_estimators=200,
        max_samples=0.6,
        max_depth=2
    )

    # Configure data module for fitting the entire dataset with classifier
    dm.set_use_triplets(False)
    dm.batch_size = -1
    # Batch is entire dataset
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))

    # Fit
    train_embeddings = model(train_batch.x)
    forest_clf.fit(train_embeddings.detach(), train_batch.y)

    # Evaluate
    val_embeddings = model(val_batch.x)
    pred = forest_clf.predict(val_embeddings.detach())

    # Visualize how many principal components are required to represent data
    # plot_top_eigenvalues(val_features)

    # Visualize the latent neighborhoods with TSNE
    tsne = TSNE(n_components=2, perplexity=50)
    manifold = tsne.fit_transform(val_embeddings.detach())
    visualize_latent_space(manifold, val_batch.y, dm.class_names(), show=True)

    # Compute and return metric
    print(f"f1: {f1_score(val_batch.y, pred, average='macro')}")
    print(f"accuracy: {accuracy_score(val_batch.y, pred)}")

    # in order to handle raise ValueError("At least one label specified must be in y_true")
    figure = ConfusionMatrixDisplay.from_predictions(val_batch.y, pred, display_labels=dm.class_names(),
                                                     cmap='Blues')
    figure.ax_.set_title('Segment Classification for AntiSaccades')
    plt.show()

    # Compute attribute importance
    attribute_importance = {
        'vertical': compute_attribute_importance(val_batch.a, pred, val_batch.y),
        'horizontal': compute_attribute_importance(1 - val_batch.a, pred, val_batch.y),
        'prosaccade': compute_attribute_importance(1 - val_batch.s, pred, val_batch.y),
        'antisaccade': compute_attribute_importance(val_batch.s, pred, val_batch.y),
        'horizontal antisaccade': compute_attribute_importance(val_batch.s * (1 - val_batch.a), pred, val_batch.y),
    }

    print(attribute_importance)


def compute_attribute_importance(attribute, pred, labels):
    return f1_score(labels, pred, average='macro', sample_weight=attribute) - \
           f1_score(labels, pred, average='macro')


if __name__ == '__main__':
    main()
