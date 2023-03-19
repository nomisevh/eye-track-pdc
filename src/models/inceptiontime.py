from typing import Union, List, cast

import torch
from prettytable import PrettyTable
from pytorch_lightning import LightningModule
from torch import nn, argsort, norm
from torch.nn import TripletMarginLoss
from torch.nn.functional import normalize
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class InceptionTimeBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual connections can be created.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 residual: bool, stride: int = 1, bottleneck_channels: int = 32,
                 kernel_size: int = 41) -> None:
        assert kernel_size > 3, "Kernel size must be strictly greater than 3"
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False, padding='same')
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        self.conv_layers = nn.Sequential(*[
            nn.Conv1d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=kernel_size_s[i],
                      stride=stride, bias=False, padding='same')
            for i in range(len(kernel_size_s))
        ])

        self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False,
                          padding='same'),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x)

        if self.use_residual:
            x = x + self.residual(org_x)
        return x


class LitInceptionTimeModel(LightningModule):
    """
    A PyTorch implementation of the InceptionTime model.
    Paper URL: https://arxiv.org/abs/1909.04939
    Code URL: https://github.com/okrasolar/pytorch-timeseries/blob/master/src/models/inception.py
    """

    def __init__(self, num_blocks: int, in_channels: int, out_channels: Union[List[int], int],
                 bottleneck_channels: Union[List[int], int], kernel_sizes: Union[List[int], int],
                 use_residuals: Union[List[bool], bool, str] = 'default',
                 num_pred_classes: int = 1, lr: float = 1e-3, wd: float = 1e-2,
                 num_semi_hard_negatives: int = None) -> None:
        """
        Attributes
        ----------
        num_blocks:
            The number of inception blocks to use. One inception block consists
            of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual
            connector
        in_channels:
            The number of input channels (i.e. input.shape[-1])
        out_channels:
            The number of "hidden channels" to use. Can be a list (for each block) or an
            int, in which case the same value will be applied to each block
        bottleneck_channels:
            The number of channels to use for the bottleneck. Can be list or int. If 0, no
            bottleneck is applied
        kernel_sizes:
            The size of the kernels to use for each inception block. Within each block, each
            of the 3 convolutional layers will have kernel size
            `[kernel_size // (2 ** i) for i in range(3)]`
        num_pred_classes:
            The number of output classes
        """
        super().__init__()

        channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels,
                                                                          num_blocks))
        bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels,
                                                                     num_blocks))
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = cast(List[bool],
                             self._expand_to_blocks(cast(Union[bool, List[bool]], use_residuals), num_blocks))

        self.blocks = nn.Sequential(*[
            InceptionTimeBlock(in_channels=channels[i], out_channels=channels[i + 1],
                               residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                               kernel_size=kernel_sizes[i])
            for i in range(num_blocks)
        ])

        # a global average pooling (i.e. mean of the time dimension) is why
        # in_features=channels[-1]
        self.linear = nn.Linear(in_features=channels[-1], out_features=num_pred_classes)
        self.out_dim = num_pred_classes

        self.loss_fn = TripletMarginLoss(margin=0.2, p=2, swap=True)
        self.lr = lr
        self.wd = wd
        self.num_semi_hard_negatives = num_semi_hard_negatives
        self.init()
        self.save_hyperparameters()

    def count_params(self, verbose: bool = False) -> int:
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        if verbose:
            print(table)
            print(f"Total Trainable Params: {total_params}")
        return total_params

    @staticmethod
    def _expand_to_blocks(value: Union[int, bool, List[int], List[bool]],
                          num_blocks: int) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.blocks(x).mean(dim=-1)  # the mean is the global average pooling
        # L2 Normalize output dimension to enforce outputs in unit hypersphere
        return normalize(self.linear(x))

    def init(self) -> 'LitInceptionTimeModel':
        def initialize_weights(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

        self.apply(initialize_weights)
        return self.train()

    def shared_step(self, batch, batch_idx, step):
        anchor, positive, negative = batch

        anchor_out = self(anchor.x)
        positive_out = self(positive.x)
        negative_out = self(negative.x)

        if step == 'train' and self.num_semi_hard_negatives is not None:
            positive_dist = norm(anchor_out - positive_out, dim=-1)
            negative_dist = norm(anchor_out - negative_out, dim=-1)
            # Select the semi-hard negatives, i.e. the hardest negatives that are still further from the anchor than the
            # positive. This is done in favour of naively choosing the hardest negatives, to avoid local minima.
            negatives_order = argsort(negative_dist)
            semi_hard_indices = negatives_order[negative_dist[negatives_order] > positive_dist[negatives_order]]

            # Use the num_semi_hard_negatives hardest semi-hard triplets.
            anchor_out = anchor_out[semi_hard_indices[:self.num_semi_hard_negatives]]
            positive_out = positive_out[semi_hard_indices[:self.num_semi_hard_negatives]]
            negative_out = negative_out[semi_hard_indices[:self.num_semi_hard_negatives]]

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
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [lr_scheduler]
