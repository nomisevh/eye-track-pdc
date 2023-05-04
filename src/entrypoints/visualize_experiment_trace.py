from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from numpy import arange
from sklearn.manifold import TSNE
from torch import repeat_interleave
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
from dataset import Signature, KIDataset
from models.inceptiontime import EndToEndInceptionTimeClassifier
from utils.const import SEED
from utils.misc import set_random_state
from utils.path import config_path, checkpoint_path


def main():
    set_random_state(SEED)

    # Load configs
    with open(config_path.joinpath('leif.yaml')) as reader:
        processor_config = load_yaml(reader, Loader=FullLoader)

    dm = KIDataModule(processor_config=processor_config,
                      use_triplets=False,
                      exclude=['vert'],
                      binary_classification=True,
                      bundle_as_experiments=True,
                      batch_size=-1)
    dm.setup('fit')
    dm.setup('test')

    model = EndToEndInceptionTimeClassifier.load_from_checkpoint(checkpoint_path.joinpath('PDC-322-epoch=149.ckpt'))
    # Freeze parameters of the encoder
    model.freeze()

    # Get the batch of the entire dataset, bundled by experiment
    val_batch_exp = next(iter(dm.test_dataloader()))

    # Get the batch of the entire dataset, flattened
    # Save the original shape of x
    original_shape = val_batch_exp.x.shape
    attributes = {'x': val_batch_exp.x.reshape(-1, val_batch_exp.x.shape[-2], val_batch_exp.x.shape[-1])}

    # Expand y, z, r, a and s to match the new shape of x
    for attr in KIDataset.SINGULAR_ATTRIBUTES:
        attributes[attr] = repeat_interleave(getattr(val_batch_exp, attr), original_shape[1], dim=0)

    val_batch = Signature(**attributes)

    val_embeddings, _ = model(val_batch.x)

    num_segments = val_batch_exp.x.shape[1]

    # Use modulo to get the index of each segment in the flattened batch, relative to every experiment
    index_in_experiment = arange(val_embeddings.shape[0]) % num_segments

    # Visualize the latent neighborhoods with TSNE
    tsne = TSNE(n_components=2, perplexity=50)
    manifold = tsne.fit_transform(val_embeddings)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=400)

    scatter = ax.scatter(manifold[:, 0], manifold[:, 1],
                         c=index_in_experiment,
                         cmap='plasma', s=40)

    cbar = plt.colorbar(scatter, ax=ax)
    ax.legend(handles=scatter.legend_elements()[0], labels=dm.class_names())
    ax.set_title('Trials by Index in Session')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    cbar.ax.set_ylabel('Index in Session (s)')
    plt.show()


if __name__ == '__main__':
    main()
