import math

import torch
from torch import nn
from torch.nn import Module, Linear


class ROCKETClassifier(Module):
    def __init__(self, rocket, output_dim=1):
        super(ROCKETClassifier, self).__init__()
        self.rocket = rocket
        # ROCKET produces two real-valued feature for every kernel. (max & ppv)
        self.classifier = Linear(rocket.n_kernels * 2, out_features=output_dim)

    def forward(self, x):
        z = self.rocket(x)
        return self.classifier(z)


class PositionalEncoding(nn.Module):
    """
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class IndividualClassifier(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(self, in_features: int = 4, num_layers: int = 4, n_classes: int = 2, **tf_kwargs):
        super().__init__()
        self.upscaler = nn.Linear(in_features, tf_kwargs['d_model'])
        self.pos_encoder = PositionalEncoding(
            d_model=tf_kwargs['d_model'],
            dropout=tf_kwargs['dropout'],
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**tf_kwargs),
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(tf_kwargs['d_model'], n_classes)
        self.d_model = tf_kwargs['d_model']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upscaler(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # return self.classifier(x[:, 0])
        return self.classifier(x.mean(dim=1))
