import numpy as np
import torch
from pytorch_lightning import Callback
from torch import sigmoid
from torchmetrics.classification import MulticlassAveragePrecision
from torchmetrics.functional.classification import multiclass_f1_score


# Only works for binary classification. Pass either logits or scores ~[0, 1].
def vote_aggregation(*, segment_logits=None, segment_scores=None, labels, aggregate_by, threshold=0.2):
    unique, inv_idx = np.unique(aggregate_by, return_inverse=True)
    patient_pred = torch.zeros(unique.shape)
    patient_label = torch.zeros(unique.shape)
    patient_probs = torch.zeros(unique.shape)
    for i, p in enumerate(unique):
        patient_mask = aggregate_by == unique[i]
        if segment_logits is not None:
            patient_segment_scores = sigmoid(torch.from_numpy(segment_logits[patient_mask])).numpy()
        elif segment_scores is not None:
            patient_segment_scores = segment_scores[patient_mask]
        else:
            raise ValueError("mandatory to pass segment scores or logits")
        mean_pred = patient_segment_scores.mean()
        patient_probs[i] = mean_pred
        patient_pred[i] = 1 if mean_pred > threshold else 0
        patient_label[i] = labels[patient_mask][0]
    return patient_pred, patient_label, patient_probs


def unweighted_binary_average_precision(preds, targets):
    multi_class_probs = torch.stack([1 - preds, preds], dim=1)
    return MulticlassAveragePrecision(num_classes=2, average='macro')(multi_class_probs, targets.long())


# Finds the best binary threshold with respect to the unweighted f1 score
def max_f1_score(pred, targets, num_thresholds=100):
    f1_scores = torch.zeros(num_thresholds)
    thresholds = torch.linspace(0, 1, num_thresholds)
    for i, threshold in enumerate(thresholds):
        probs = (pred >= threshold).long()
        f1_scores[i] = multiclass_f1_score(probs, targets.long(), num_classes=2, average='macro')

    return f1_scores.max().item(), thresholds[f1_scores.argmax()].item()


class ValidationMetricCallback(Callback):
    def __init__(self, metric, mode='max'):
        super().__init__()
        self.metric = metric
        self.mode = mode
        self.best = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.mode == 'max':
            current = trainer.logged_metrics[self.metric]
            if self.best is None or current > self.best:
                self.best = current
        else:
            raise NotImplementedError()
