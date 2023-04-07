import numpy as np
import torch
from torch import sigmoid
from torchmetrics.classification import MulticlassAveragePrecision


# Only works for binary classification. Pass either logits or scores ~[0, 1].
def vote_aggregation(*, segment_logits=None, segment_scores=None, labels, aggregate_by, threshold=0.2):
    unique, inv_idx = np.unique(aggregate_by, return_inverse=True)
    patient_pred = np.zeros(unique.shape)
    patient_label = np.zeros(unique.shape)
    patient_probs = np.zeros(unique.shape)
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
