import math

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, zeros
from torch.nn import Dropout, Linear, TransformerEncoderLayer, TransformerEncoder, BCEWithLogitsLoss, Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import f1_score
from torchmetrics.functional.classification import binary_accuracy


class PositionalEncoding(Module):
    """
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = Dropout(p=dropout)

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


class TransformerClassifier(Module):

    def __init__(self, *, in_features: int = 4, num_layers: int = 4, n_classes: int = 2, **tf_kwargs):
        super().__init__()
        self.projector = Linear(in_features, tf_kwargs['d_model'])
        self.pos_encoder = PositionalEncoding(
            d_model=tf_kwargs['d_model'],
            dropout=tf_kwargs['dropout'],
        )
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(**tf_kwargs),
            num_layers=num_layers,
        )
        self.classifier = Linear(tf_kwargs['d_model'], n_classes)
        self.d_model = tf_kwargs['d_model']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project every item in the sequence to the dimensionality of the transformer encoder
        x = self.projector(x)
        # Apply positional encoding
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # return self.decoder(x[:, 0])
        return self.classifier(x.mean(dim=1)).squeeze()


class LitTimeSeriesClassifier(LightningModule):
    """
    Classifies a trial based on the list of segments of the trial time series.
    """

    def __init__(self, encoder, decoder, feature_dim, lr, wd):
        super(LitTimeSeriesClassifier, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_dim = feature_dim
        self.lr = lr
        self.wd = wd
        self.loss_fn = BCEWithLogitsLoss()
        self.save_hyperparameters(ignore=['encoder', 'decoder'])

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: A tensor of shape (N, L, M, T), where N is the batch size and L is the number of segments in the trial
        :return: A tensor of shape (N) with logits
        """
        n, l, m, t = x.shape
        tokens = zeros(n, l, self.feature_dim).to(self.device)
        for i in range(n):
            # Pass every segment through the feature extractor
            tokens[i] = self.encoder(x[i])

        return self.decoder(tokens)

    def training_step(self, batch, batch_idx):
        logits = self(batch.x)
        loss = self.loss_fn(logits, batch.y)
        self.log_dict({'train_loss': loss,
                       'train_accuracy': binary_accuracy(logits, batch.y.long()),
                       'train_f1': f1_score(logits, batch.y.long(), task='binary', average='macro')}, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch.x)
        loss = self.loss_fn(logits, batch.y)
        self.log_dict({'val_loss': loss,
                       'val_accuracy': binary_accuracy(logits, batch.y.long()),
                       'val_f1': f1_score(logits, batch.y.long(), task='binary', average='macro')})
        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch.x)
        loss = self.loss_fn(logits, batch.y)
        self.log_dict({'test_loss': loss,
                       'test_accuracy': binary_accuracy(logits, batch.y.long()),
                       'test_f1': f1_score(logits, batch.y.long(), task='binary', average='macro')})
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [lr_scheduler]
