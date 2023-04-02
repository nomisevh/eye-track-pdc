from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
from models.inceptiontime import LitInceptionTime
from utils.const import SEED
from utils.misc import set_random_state
from utils.path import config_path, log_path, checkpoint_path


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

    # model = LitInceptionTime.load_from_checkpoint(checkpoint_path.joinpath('epoch=49-step=200-v1.ckpt'))
    model = LitInceptionTime.load_from_checkpoint(checkpoint_callback.best_model_path)

    dm.set_use_triplets(False)
    dm.batch_size = -1

    # For evaluation metrics
    labels = ['HC', 'PD']

    # trainer.fit(classifier, datamodule=dm)

    # Freeze parameters of the encoder
    model.freeze()

    # Initialize classifiers
    forest_clf = RandomForestClassifier(
        n_jobs=1,
        n_estimators=200,
        max_samples=0.82,
        max_depth=2
    )
    svm_clf = SVC(
        C=1,
        kernel='rbf',
        gamma='scale',
        probability=True
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

    # Visualize how many principal components are required to represent data
    # plot_top_eigenvalues(val_features)

    # Visualize the latent neighborhoods with TSNE
    # tsne = TSNE(n_components=2, perplexity=50)
    # visualize_latent_space(tsne, val_features, val_batch, labels)

    # Evaluate
    val_embeddings = model(val_batch.x)
    pred = forest_clf.predict(val_embeddings.detach())

    # The SVMClassifier maps the targets to {-1, 1}, but our labels are {0, 1}
    pred[pred < 0] = 0

    # Compute and return metric
    print(f"f1: {f1_score(val_batch.y, pred, average='macro')}")
    print(f"accuracy: {accuracy_score(val_batch.y, pred)}")


if __name__ == '__main__':
    main()
