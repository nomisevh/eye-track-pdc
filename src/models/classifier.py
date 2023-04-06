from pytorch_lightning import LightningModule
from torch import nn, Tensor, stack
from torch.nn import RNN
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics.functional import f1_score
from torchmetrics.functional.classification import binary_accuracy

from utils.misc import initialize_weights


class LitLinearClassifier(LightningModule):

    def __init__(self, feature_dim: int, num_classes: int, lr: float, wd: float, feature_extractor: nn.Module):
        super().__init__()

        self.feature_extractor = feature_extractor
        # self.classifier = nn.Linear(feature_dim, num_classes)
        self.hidden_dim = 64
        self.classifier = nn.Sequential(*[nn.Linear(feature_dim, self.hidden_dim),
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
        self.lr = lr
        self.wd = wd
        self.clf_loss = nn.BCEWithLogitsLoss()
        self.apply(initialize_weights)
        self.save_hyperparameters(ignore=['feature_extractor'])

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits.squeeze()

    def shared_step(self, batch, batch_idx, step):
        logits = self(batch.x)
        loss = self.clf_loss(logits, batch.y)
        probs = nn.functional.sigmoid(logits)

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
        logits = self(batch.x)
        probs = nn.functional.sigmoid(logits)

        self.log_dict({
            # The accuracy for the anchor predictions
            f'test_accuracy': binary_accuracy(probs, batch.y),
            # The F1 score for the anchor predictions
            f'test_f1': f1_score(probs, batch.y, task='binary', average='macro')},
        )

        # Return the predictions for plotting confusion matrix
        return probs.round()

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        logits = self(batch.x).squeeze()
        probs = nn.functional.sigmoid(logits)
        return probs.round()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

        warmup_epochs = self.trainer.max_epochs // 20
        # Cosine annealing with warmup, to allow use of learning rate scaling rule, see Goyal et al. 2018
        lr_scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs - warmup_epochs)
        ], milestones=[warmup_epochs])

        return [optimizer], [lr_scheduler]


class LitAggregateClassifier(LightningModule):

    def __init__(self, num_segments: int, feature_dim: int, num_classes: int, lr: float, wd: float,
                 feature_extractor: nn.Module):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.feature_extractor.freeze()
        self.hidden_dim = 4
        self.classifier = nn.Linear(feature_dim * num_segments, num_classes)

        # A simple classifier with 1 hidden layer
        self.classifier = nn.Sequential(*[nn.Linear(feature_dim * num_segments, self.hidden_dim),
                                          nn.Dropout(p=0.5),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, num_classes),
                                          ])

        self.rnn = RNN(input_size=feature_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True,
                       nonlinearity='relu', dropout=0.2)
        # self.lstm = LSTM(input_size=feature_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        # self.classifier = nn.Sequential(*[nn.Linear(feature_dim * num_segments, self.hidden_dim),
        #                                   nn.Dropout(),
        #                                   nn.ReLU(),
        #                                   nn.Linear(self.hidden_dim, self.hidden_dim),
        #                                   nn.Dropout(),
        #                                   nn.ReLU(),
        #                                   nn.Linear(self.hidden_dim, self.hidden_dim),
        #                                   nn.Dropout(),
        #                                   nn.ReLU(),
        #                                   nn.Linear(self.hidden_dim, num_classes),
        #                                   ])

        self.lr = lr
        self.wd = wd
        self.clf_loss = nn.BCEWithLogitsLoss()
        self.apply(initialize_weights)
        self.save_hyperparameters(ignore=['feature_extractor'])

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        """
        :param x: shape (N, L, M, T)
        :return:
        """
        n, l, m, t = x.shape
        features = []
        # For every experiment in the batch, extract the features from the segments
        for i in range(n):
            features.append(self.feature_extractor(x[i]))

        features = stack(features)
        # Concatenate the features from segments for every experiment
        features = features.view(n, -1)

        # outputs, (hn, cn) = self.lstm(features)
        # outputs, _ = self.rnn(features)

        # logits = self.classifier(outputs[:, -1, :])
        logits = self.classifier(features)
        return logits.squeeze()

    def shared_step(self, batch, batch_idx, step):
        logits = self(batch.x)
        loss = self.clf_loss(logits, batch.y)
        probs = nn.functional.sigmoid(logits)

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

        warmup_epochs = self.trainer.max_epochs // 20
        # Cosine annealing with warmup, to allow use of learning rate scaling rule, see Goyal et al. 2018
        lr_scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs - warmup_epochs)
        ], milestones=[warmup_epochs])

        return [optimizer], [lr_scheduler]
