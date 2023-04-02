from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
from models.inceptiontime import LitInceptionTime, LitAggregateClassifier
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

    # Bundle as experiments before splitting into train/validation sets to avoid data leakage.
    dm = KIDataModule(processor_config=processor_config,
                      use_triplets=True,
                      binary_classification=True,
                      bundle_as_experiments=True,
                      batch_size=256)
    dm.setup('fit')
    # Flatten the experiment dimension before training InceptionTime on the segments.
    dm.flatten()

    model = LitInceptionTime(**inception_config)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor='val_loss', every_n_epochs=1)
    trainer = Trainer(accelerator='auto',
                      max_epochs=100,
                      default_root_dir=log_path,
                      log_every_n_steps=1,
                      logger=NeptuneLogger(log_model_checkpoints=False, **neptune_config),
                      callbacks=[checkpoint_callback])

    # trainer.fit(model, datamodule=dm)

    trainer = Trainer(accelerator='auto',
                      max_epochs=200,
                      default_root_dir=log_path,
                      log_every_n_steps=1,
                      logger=NeptuneLogger(log_model_checkpoints=False, **neptune_config),
                      callbacks=[checkpoint_callback])
    # model = LitInceptionTime.load_from_checkpoint(checkpoint_callback.best_model_path)
    model = LitInceptionTime.load_from_checkpoint(checkpoint_path.joinpath('epoch=49-step=200-v3.ckpt'))

    # Batch the segments by experiment before training the aggregate classifier.
    dm.batch()
    dm.set_use_triplets(False)
    dm.batch_size = 8

    classifier = LitAggregateClassifier(num_segments=dm.train_ds.x.shape[1],
                                        feature_dim=model.out_dim,
                                        num_classes=1,
                                        lr=0.05,
                                        wd=0.001,
                                        feature_extractor=model)

    # transformer = TransformerClassifier(in_features=model.out_dim,
    #                                     num_layers=4,
    #                                     n_classes=1,
    #                                     d_model=128,
    #                                     nhead=4,
    #                                     dim_feedforward=256,
    #                                     dropout=0.1)

    # classifier = LitTimeSeriesClassifier(encoder=model,
    #                                      decoder=lstm,
    #                                      feature_dim=model.out_dim,
    #                                      lr=0.0001,
    #                                      wd=0.01)

    trainer.fit(classifier, datamodule=dm)


def fit_and_predict_clf(dm: KIDataModule, checkpoint_filename):
    inception_time = LitInceptionTime.load_from_checkpoint(checkpoint_path.joinpath(checkpoint_filename))

    # Freeze parameters of the encoder
    inception_time.freeze()

    # Initialize classifiers
    forest_clf = RandomForestClassifier(
        n_jobs=1,
        n_estimators=200,
        max_samples=0.826,
        max_depth=2
    )

    # Configure data module for fitting the entire dataset with classifier
    dm.set_use_triplets(False)
    dm.batch_size = -1
    # Batch is entire dataset
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))

    # Fit
    train_embeddings = inception_time(train_batch.x)
    forest_clf.fit(train_embeddings.detach(), train_batch.y)

    # Evaluate
    val_embeddings = inception_time(val_batch.x)
    pred = forest_clf.predict(val_embeddings.detach())

    # Compute and return metric
    return f1_score(val_batch.y, pred, average='macro')


if __name__ == '__main__':
    main()
