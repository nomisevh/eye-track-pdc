from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from numpy import linspace, arange, in1d
from sklearn.manifold import TSNE
from torch import repeat_interleave
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
from dataset import Signature, KIDataset
from models.inceptiontime import LitInceptionTime
from utils.const import SEED
from utils.misc import set_random_state
from utils.path import config_path, checkpoint_path
from utils.visualize import visualize_latent_space


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

    model = LitInceptionTime.load_from_checkpoint(checkpoint_path.joinpath('epoch=459-step=1840.ckpt'))
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

    val_embeddings = model(val_batch.x)

    experiment = 15
    num_segments = val_batch_exp.x.shape[1]
    indices = arange(experiment * num_segments, (experiment + 1) * num_segments).astype(int)
    inverse_indices = in1d(arange(val_embeddings.shape[0]), indices, invert=True)

    # Visualize the latent neighborhoods with TSNE
    tsne = TSNE(n_components=2, perplexity=15)
    manifold = tsne.fit_transform(val_embeddings)
    # Visualize the latent space for all segments except the ones belonging to the experiment
    fig, ax = visualize_latent_space(manifold[inverse_indices], val_batch.y[inverse_indices], dm.class_names(),
                                     show=False)

    # Plot the segments for the experiment to visualize the latent trace of the segments
    manifold_exp = manifold[indices]
    # Manually set colors to have different opacity depending on the time of the segment
    color = [to_rgba('crimson', alpha) for alpha in linspace(1, 0.2, num_segments)]
    ax.scatter(manifold_exp[:, 0], manifold_exp[:, 1], c=color, s=40, label=f'Experiment {experiment}')
    ax.set_title('PD Diseased Experiment')
    plt.show()
    print(f'label: {val_batch_exp.y[experiment]}')
    print(f'group: {val_batch_exp.g[experiment]}')
    print(f'a: {val_batch_exp.a[experiment]}')
    print(f's: {val_batch_exp.s[experiment]}')


if __name__ == '__main__':
    main()
