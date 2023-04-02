from numpy import argsort
from pytorch_lightning import LightningModule
from torch import nn, Tensor, concatenate, norm
from torch.nn import ModuleList, TripletMarginLoss
from torch.nn.functional import relu, normalize
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR

from utils.misc import initialize_weights


class InceptionModule(nn.Module):
    NUM_FILTER_SETS = 3

    def __init__(self, in_dim: int, hidden_dim: int, bottleneck_dim: int, base_kernel_size: int,
                 residual: bool):
        min_base_kernel_size = 2 ** (self.NUM_FILTER_SETS - 1)
        assert base_kernel_size >= min_base_kernel_size, f'base kernel size must be {min_base_kernel_size} or greater'
        super().__init__()

        # The outputs from the filter sets will be concatenated feature-wise along with the parallel low pass filter.
        out_dim = hidden_dim * (self.NUM_FILTER_SETS + 1)
        filter_in_dim = in_dim
        if bottleneck_dim > 0 and in_dim > 1:
            self.bottleneck = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1, bias=False, padding='same')
            filter_in_dim = bottleneck_dim

        kernel_sizes = [base_kernel_size // (2 ** i) for i in range(self.NUM_FILTER_SETS)]

        self.filter_sets = ModuleList([
            nn.Conv1d(filter_in_dim, hidden_dim, kernel_size=ks, padding='same', bias=False) for ks in kernel_sizes])

        self.bn = nn.BatchNorm1d(out_dim)

        self.parallel_low_pass_filter = nn.Sequential(*[
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1, padding='same', bias=False)
        ])

        # Not sure if residual should be used at every module, but from the paper it doesn't seem to have a significant
        # effect anyway.
        if residual:
            self.residual = nn.Sequential(*[
                nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False, padding='same'),
                # Not sure if it's important that the residual has its own batch norm. Will keep it just in case.
                nn.BatchNorm1d(out_dim),
            ])

    def forward(self, x: Tensor) -> Tensor:
        org_x = x
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        filter_outputs = []
        for filter_set in self.filter_sets:
            filter_outputs.append(filter_set(x))

        filter_outputs.append(self.parallel_low_pass_filter(org_x))

        x = concatenate(filter_outputs, dim=1)
        x = self.bn(x)

        if self.residual is not None:
            x = x + self.residual(org_x)

        return relu(x)


class LitInceptionTime(LightningModule):

    def __init__(self, depth: int, in_dim: int, hidden_dim: int, bottleneck_dim: int, base_kernel_size: int,
                 use_residuals: bool, lr: float = 1e-4, wd: float = 1e-3, num_semi_hard_negatives: int = None,
                 anchor_swap: bool = True, triplet_margin: float = 0.2, normalize_output: bool = False):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.num_semi_hard_negatives = num_semi_hard_negatives
        self.normalize_output = normalize_output

        # The outputs from the filter sets will be concatenated feature-wise along with the parallel low pass filter.
        self.out_dim = (InceptionModule.NUM_FILTER_SETS + 1) * hidden_dim
        self.inception_modules = nn.Sequential(*[InceptionModule(in_dim=in_dim if i == 0 else self.out_dim,
                                                                 hidden_dim=hidden_dim,
                                                                 bottleneck_dim=bottleneck_dim,
                                                                 base_kernel_size=base_kernel_size,
                                                                 residual=use_residuals) for i in range(depth)])
        self.loss_fn = TripletMarginLoss(margin=triplet_margin, p=2, swap=anchor_swap)

        self.apply(initialize_weights)
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        # Pass through inception modules and do global average pooling over the final MTS --> (N, hidden_dim)
        x = self.inception_modules(x).mean(dim=-1)

        return normalize(x) if self.normalize_output else x

    def shared_step(self, batch, batch_idx, step):
        anchor, positive, negative = batch

        anchor_out = self(anchor.x)
        positive_out = self(positive.x)
        negative_out = self(negative.x)

        # Validation is performed on the entire batch
        if step == 'train' and self.num_semi_hard_negatives is not None:
            anchor_out, positive_out, negative_out = compute_semi_hard_negatives(anchor_out, positive_out, negative_out,
                                                                                 self.num_semi_hard_negatives)

        loss = self.loss_fn(anchor_out, positive_out, negative_out)
        self.log(f'{step}_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, *args):
        return self.shared_step(*args, step='train')

    def validation_step(self, *args):
        return self.shared_step(*args, step='val')

    def test_step(self, *args):
        return self.shared_step(*args, step='test')

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

        warmup_epochs = self.trainer.max_epochs // 20
        # Cosine annealing with warmup, to allow use of learning rate scaling rule, see Goyal et al. 2018
        lr_scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs - warmup_epochs)
        ], milestones=[warmup_epochs])

        return [optimizer], [lr_scheduler]


def compute_semi_hard_negatives(anchor_embeddings, positive_embeddings, negative_embeddings, num_semi_hard_negatives):
    positive_dist = norm(anchor_embeddings - positive_embeddings, dim=-1)
    negative_dist = norm(anchor_embeddings - negative_embeddings, dim=-1)
    # Select the semi-hard negatives, i.e. the hardest negatives that are still further from the anchor than the
    # positive. This is done in favour of naively choosing the hardest negatives, to avoid local minima.
    negatives_order = argsort(negative_dist)
    semi_hard_indices = negatives_order[negative_dist[negatives_order] > positive_dist[negatives_order]]

    # Use the num_semi_hard_negatives hardest semi-hard triplets.
    anchor_embeddings = anchor_embeddings[semi_hard_indices[:num_semi_hard_negatives]]
    positive_embeddings = positive_embeddings[semi_hard_indices[:num_semi_hard_negatives]]
    negative_embeddings = negative_embeddings[semi_hard_indices[:num_semi_hard_negatives]]

    return anchor_embeddings, positive_embeddings, negative_embeddings
