from numpy import zeros
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from torch import arange
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
from models.inceptiontime import LitInceptionTime
from utils.const import SEED
from utils.misc import set_random_state
from utils.path import config_path, log_path, checkpoint_path

N_MODELS = 5
SEEDS = arange(N_MODELS)


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
                      use_triplets=False,
                      binary_classification=True,
                      batch_size=-1)
    dm.setup('fit')

    # Prepare batch that contains the entire dataset
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))

    # Prepare datamodule for training feature extractor
    dm.set_use_triplets(True)
    dm.batch_size = 256

    # Train N_MODELS models
    models = []
    classifiers = []
    val_predictions = zeros((N_MODELS, len(dm.val_ds)))
    for i, seed in enumerate(SEEDS):
        set_random_state(seed)
        model = LitInceptionTime(**inception_config)
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor='val_loss', every_n_epochs=1)
        trainer = Trainer(accelerator='auto',
                          max_epochs=100,
                          default_root_dir=log_path,
                          log_every_n_steps=1,
                          logger=NeptuneLogger(log_model_checkpoints=False, **neptune_config),
                          callbacks=[checkpoint_callback])

        trainer.fit(model, datamodule=dm)
        model = LitInceptionTime.load_from_checkpoint(checkpoint_callback.best_model_path)
        model.freeze()
        models.append(model)

        # Initialize classifiers
        forest_clf = RandomForestClassifier(
            n_jobs=1,
            n_estimators=200,
            max_samples=0.6,
            max_depth=2
        )

        # Fit
        train_embeddings = model(train_batch.x)
        forest_clf.fit(train_embeddings.detach(), train_batch.y)
        classifiers.append(forest_clf)

        # Evaluate
        val_embeddings = model(val_batch.x)
        pred = forest_clf.predict(val_embeddings.detach())
        val_predictions[i] = pred

    # Make final predictions by taking the most frequent prediction
    pred = zeros(val_predictions.shape[1])
    for i in range(val_predictions.shape[1]):
        pred[i] = (int(round(val_predictions[:, i].mean())))

    # Compute and print metrics
    print(f"f1: {f1_score(val_batch.y, pred, average='macro')}")
    print(f"accuracy: {accuracy_score(val_batch.y, pred)}")


if __name__ == '__main__':
    main()
