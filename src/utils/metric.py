import numpy as np
import torch
from torch import sigmoid


# Only works for binary classification. Pass either logits or scores ~[0, 1].
def patient_soft_accuracy(*, segment_logits=None, segment_scores=None, y, z, threshold=0.2):
    unique, inv_idx = np.unique(z, return_inverse=True)
    patient_pred = np.zeros(unique.shape)
    patient_label = np.zeros(unique.shape)
    for i, p in enumerate(unique):
        patient_mask = z == unique[i]
        if segment_logits is not None:
            patient_segment_scores = sigmoid(torch.from_numpy(segment_logits[patient_mask])).numpy()
        elif segment_scores is not None:
            patient_segment_scores = segment_scores[patient_mask]
        else:
            raise ValueError("mandatory to pass segment scores or logits")
        mean_pred = np.mean(patient_segment_scores, axis=-1)
        patient_pred[i] = 1 if mean_pred > threshold else 0
        patient_label[i] = y[patient_mask][0]
    patient_acc = (patient_pred == patient_label).mean().item()
    return patient_pred, patient_label, patient_acc
