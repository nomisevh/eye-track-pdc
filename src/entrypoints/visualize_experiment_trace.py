from matplotlib import pyplot as plt
from numpy import linspace
from sklearn.manifold import TSNE
from torch import ones
from yaml import load as load_yaml, FullLoader

from datamodule import KIDataModule
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
                      binary_classification=True,
                      bundle_as_experiments=True,
                      batch_size=-1)
    dm.setup('fit')

    model = LitInceptionTime.load_from_checkpoint(checkpoint_path.joinpath('use_this.ckpt'))
    # Freeze parameters of the encoder
    model.freeze()

    # Get the batch of the entire dataset, bundled by experiment
    val_batch_exp = next(iter(dm.val_dataloader()))

    dm.flatten()
    # Get the batch of the entire dataset, flattened
    val_batch = next(iter(dm.val_dataloader()))

    val_embeddings = model(val_batch.x)

    # Visualize the latent neighborhoods with TSNE
    tsne = TSNE(n_components=2, perplexity=50)
    manifold = tsne.fit_transform(val_embeddings)
    fig, ax = visualize_latent_space(manifold, val_batch, dm.class_names())

    # Re-plot the segments for an experiment to visualize the latent trace of the experiment
    # Embed the segments belonging to one of the experiments
    experiment = 0
    val_embeddings_exp = model(val_batch_exp.x[experiment])
    manifold_exp = tsne.fit_transform(val_embeddings_exp)
    # Get the labels for the segments in the experiment and create a tensor with the label for each segment
    num_segments = val_embeddings_exp.shape[0]
    labels = ones(num_segments) * val_batch_exp.y[experiment]

    ax.plot(manifold_exp, labels, 'o', color='red', alpha=linspace(1, 0.1, num_segments))
    plt.show()


if __name__ == '__main__':
    main()
