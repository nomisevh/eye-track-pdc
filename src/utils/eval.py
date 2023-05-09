from torchmetrics.functional.classification import multiclass_f1_score, binary_accuracy

from model_selection import get_attribute_power
from utils.metric import vote_aggregation, max_f1_score
from utils.visualize import plot_confusion_matrix, plot_latent_neighborhood


def evaluate(batch, trial_probs, features, class_names, model_name):
    # Aggregate predictions to the subject level
    _, test_subject_labels, test_subject_probs = vote_aggregation(segment_scores=trial_probs, labels=batch.y,
                                                                  aggregate_by=batch.z)

    # Find the best subject threshold with respect to the unweighted f1 score on the test set
    _, test_subject_threshold = max_f1_score(test_subject_probs, test_subject_labels)

    # Compute unweighted F1 and accuracy for trials and subjects on the test set
    test_trial_uf1_score = multiclass_f1_score((trial_probs >= 0.5).long(), batch.y.long(), num_classes=2,
                                               average='macro')
    test_subject_uf1_score_best = multiclass_f1_score((test_subject_probs >= test_subject_threshold).long(),  # noqa
                                                      test_subject_labels.long(), num_classes=2, average='macro')

    # Report the results on trial level
    print(f"test trial uF1: {test_trial_uf1_score :.4f}")
    print(f"test trial accuracy: {binary_accuracy(trial_probs, batch.y):.2%}"
          f" with threshold 0.5")

    # Report the results on subject level with the best threshold found on the test set
    print(f"test subject uF1 (test threshold): {test_subject_uf1_score_best:.4f}")
    print(f"test subject accuracy (test threshold): "
          f"{binary_accuracy(test_subject_probs, test_subject_labels, threshold=test_subject_threshold):.2%}"
          f" with threshold {test_subject_threshold:.2f}")

    # Perform attribute-based subgroup evaluation
    attribute_power = get_attribute_power(batch, trial_probs)
    attribute_power = {k: f'{v:.2%}' for k, v in attribute_power.items()}
    print(f'Attribute power: {attribute_power}')

    # Plot confusion matrix for trial-level classification
    plot_confusion_matrix(batch.y, (trial_probs >= 0.5).int(), class_names,
                          title=f'Trial Classification with {model_name}',
                          show=True)

    # Plot confusion matrix for subject-level classification
    plot_confusion_matrix(test_subject_labels, (test_subject_probs >= test_subject_threshold).int(), class_names,
                          # noqa
                          title=f'Subject Classification with {model_name}',
                          show=True)

    # Visualize the latent neighborhoods with TSNE
    plot_latent_neighborhood(features, batch, class_names, show=True)
