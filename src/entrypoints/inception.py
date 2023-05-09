from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from torch.nn.functional import sigmoid
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
from models.inceptiontime import EndToEndInceptionTimeClassifier
from utils.const import SEED
from utils.eval import evaluate
from utils.misc import set_random_state, ignore_warnings
from utils.path import config_path, log_path, checkpoint_path

TAGS = ['Final Results', 'Main Results v3']
CHECKPOINT_FILENAME = 'PDC-378-epoch=18.ckpt'


def main(seed, checkpoint_filename=None):
    ignore_warnings(PossibleUserWarning, FutureWarning)
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

    if checkpoint_filename is not None:
        model = EndToEndInceptionTimeClassifier.load_from_checkpoint(checkpoint_path.joinpath(checkpoint_filename))
    else:
        model = EndToEndInceptionTimeClassifier(num_classes=1, seed=seed, **inception_config)
        trainer.fit(model, datamodule=dm)
        model = EndToEndInceptionTimeClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)

    # Perform inference on the entire test set.
    dm.batch_size = -1
    test_batch = next(iter(dm.test_dataloader()))

    # Make trial-level predictions
    features, logits = model(test_batch.x)
    probs = sigmoid(logits.squeeze())

    evaluate(test_batch, probs, features, dm.class_names(), model_name='InceptionTime')


if __name__ == '__main__':
    main(SEED, CHECKPOINT_FILENAME)

    # for seed in [42, 1337, 9000, 1, 2]:
    #    main(seed)
