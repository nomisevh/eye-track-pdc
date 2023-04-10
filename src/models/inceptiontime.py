from numpy import argsort
from pytorch_lightning import LightningModule
from torch import nn, Tensor, concatenate, norm, sigmoid
from torch.nn import ModuleList, TripletMarginLoss, BCEWithLogitsLoss
from torch.nn.functional import relu, normalize
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
from torchmetrics.functional.classification import binary_accuracy, multiclass_f1_score

from utils.metric import unweighted_binary_average_precision
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


class EndToEndInceptionTimeClassifier(LightningModule):

    def __init__(self, lr: float = 1e-4, wd: float = 1e-3, num_classes: int = 1, triplet_loss: bool = True,
                 seed: int = None, linear_clf: bool = False, **kwargs):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.triplet_loss = triplet_loss
        self.inception_time = LitInceptionTime(**kwargs)
        self.hidden_dim = 64
        # Todo: this should not be a boolean parameter.
        if linear_clf:
            self.classifier = nn.Linear(self.inception_time.out_dim, num_classes)
        else:
            self.classifier = nn.Sequential(*[nn.Linear(self.inception_time.out_dim, self.hidden_dim),
                                              nn.Dropout(),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_dim, self.hidden_dim),
                                              nn.Dropout(),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_dim, self.hidden_dim),
                                              nn.Dropout(),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_dim, num_classes),
                                              ])

        self.triplet_loss_fn = self.inception_time.loss_fn
        self.clf_loss = BCEWithLogitsLoss()

        self.apply(initialize_weights)
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        features = self.inception_time.forward(x)
        logits = self.classifier(features)
        return features, logits.squeeze()

    def shared_step(self, batch, batch_idx, step):
        anchor, positive, negative = batch

        anchor_out, anchor_logits = self(anchor.x)
        positive_out, positive_logits = self(positive.x)
        negative_out, negative_logits = self(negative.x)

        if step == 'train' and self.inception_time.num_semi_hard_negatives is not None:
            anchor_out, positive_out, negative_out = \
                compute_semi_hard_negatives(anchor_out, positive_out, negative_out,
                                            self.inception_time.num_semi_hard_negatives)

        triplet_loss = self.inception_time.loss_fn(anchor_out, positive_out, negative_out)
        bce_losses = [self.clf_loss(pred, item.y) for item, pred in
                      zip(batch, [anchor_logits, positive_logits, negative_logits])]
        # Uniform weighting of losses (incl. triplet loss)
        avg_bce_loss = sum(bce_losses) / 3

        if self.triplet_loss:
            total_loss = sum([triplet_loss, avg_bce_loss]) / 2
        else:
            total_loss = avg_bce_loss

        anchor_probs = sigmoid(anchor_logits)

        self.log_dict({
            # The combined loss
            f'{step}_loss': total_loss,
            # The triplet loss
            f'{step}_triplet_loss': triplet_loss,
            # The individual BCE losses for the anchor, positive, and negative respectively
            **{f'{step}_{item}_bce_loss': loss for item, loss in zip(['anchor', 'pos', 'neg'], bce_losses)},
            # The accuracy for the anchor predictions
            f'{step}_accuracy': binary_accuracy(anchor_probs, anchor.y),
            # The F1 score for the anchor predictions
            f'{step}_f1': multiclass_f1_score(anchor_probs, anchor.y.long(), num_classes=2, average='macro'),
            # The unweighted average precision for the anchor predictions
            f'{step}_uap': unweighted_binary_average_precision(anchor_probs, anchor.y)},
            on_epoch=True)

        return total_loss

    def training_step(self, *args):
        return self.shared_step(*args, step='train')

    def validation_step(self, *args):
        return self.shared_step(*args, step='val')

    def test_step(self, batch, batch_idx):
        _, logits = self(batch.x)

        probs = sigmoid(logits.squeeze())

        self.log_dict({
            # The accuracy for the anchor predictions
            f'test_accuracy': binary_accuracy(probs, batch.y),
            # The F1 score for the anchor predictions
            f'test_f1': multiclass_f1_score(probs, batch.y.long(), num_classes=2, average='macro')},
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        _, logits = self(batch.x)
        probs = nn.functional.sigmoid(logits.squeeze())
        return probs

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

        warmup_epochs = self.trainer.max_epochs // 20
        # Cosine annealing with warmup, to allow use of learning rate scaling rule, see Goyal et al. 2018
        lr_scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs - warmup_epochs)
        ], milestones=[warmup_epochs])

        return [optimizer], [lr_scheduler]
