import math

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, zeros, sigmoid, nn
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


class PrependClassToken(nn.Module):
    """
    PrependClassToken Class:
    Module to prepend a learnable class token to every sequence in batch.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # The class token does not carry any information in itself. The hidden state corresponding to this token at the
        # end of the transformer will be inferred by all other tokens in the sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: dimensions [batch_size, seq_len, embedding_dim]
        :return: x prepended with class token [batch_size, seq_len+1, embedding_dim]
        """
        return torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)


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
        self.prepend_cls_token = PrependClassToken(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project every item in the sequence to the dimensionality of the transformer encoder
        x = self.projector(x)
        # Prepend a class token to the sequence
        x = self.prepend_cls_token(x)
        # Apply positional encoding
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # return self.decoder(x[:, 0])
        return self.classifier(x[:, 0]).squeeze()


class LitTimeSeriesClassifier(LightningModule):
    """
    Classifies a trial based on the list of segments of the trial time series.
    """

    def __init__(self, encoder, decoder, feature_dim, lr, wd):
        super(LitTimeSeriesClassifier, self).__init__()
        self.encoder = encoder
        self.encoder.freeze()
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

    def shared_step(self, batch, batch_idx, step):
        logits = self(batch.x)
        loss = self.loss_fn(logits, batch.y)
        probs = sigmoid(logits)

        self.log_dict({
            f'{step}_loss': loss,
            # The accuracy for the predictions
            f'{step}_accuracy': binary_accuracy(probs, batch.y),
            # The F1 score for the predictions
            f'{step}_f1': f1_score(probs, batch.y, task='binary', average='macro')},
            on_epoch=True)

        return loss

    def training_step(self, *args):
        return self.shared_step(*args, step='train')

    def validation_step(self, *args):
        return self.shared_step(*args, step='val')

    def test_step(self, batch, batch_idx):
        _, logits = self(batch)
        probs = nn.functional.sigmoid(logits)

        self.log_dict({
            # The accuracy for the anchor predictions
            f'test_accuracy': binary_accuracy(probs, batch.y),
            # The F1 score for the anchor predictions
            f'test_f1': f1_score(probs, batch.y, task='binary', average='macro')},
        )

        # Return the predictions for plotting confusion matrix
        return probs.round()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [lr_scheduler]
