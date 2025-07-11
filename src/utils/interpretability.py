"""
This file contains utilities related to the feature selection and interpretability of the ROCKET model.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import (RidgeClassifierCV, RidgeClassifier)
from scipy.signal import welch

# from sktime.transformations.panel.rocket import (
#     Rocket,
#     MiniRocket,
#     MultiRocket
# )

# def train_full_rocket_model(X_train,X_test,y_train,y_test,num_kernels:int = 10000, model_type:str = "rocket", verbose = False):
#     """
#     Applies rocket transformation on time series, fits ridge classifier and evaluates results.

#     Parameters
#     ----------
#     X_train: numpy array
#         Training time series, a 3d array of shape (# instances, # dimensions, # timesteps). When used with univariate TS,
#         make sure you transform the 2d to 3d by expanding dimensions
#     X_test: numpy array
#         Test time series, a 3d array of shape (# instances, # dimensions, # timesteps). When used with univariate TS,
#         make sure you transform the 2d to 3d by expanding dimensions
#     y_train: numpy array
#         Training labels
#     y_test: numpy array
#         Test labels
#     num_kernels: int
#         Number of kernels to use
#     model_type: str
#         Type of rocket transformation to use: rocekt, minirocket or multirocket
#     verbose: bool
#         If true, prints the accuraccy results for training and test set

#     Returns
#     -------
#     rocket_model: sktime transformation object
#         ROCKET transformation module
#     classifier: sklearn model
#         Fitted ridge classifier
#     X_train_scaled_transform: numpy array
#         Training features matrix, a 2d array of shape (# instances, # dimensions)
#     X_test_scaled_transform: numpy array
#         Test features matrix, a 2d array of shape (# instances, # dimensions)
#     full_model_score_train: float
#         Balanced accuraccy of the classifier on the training set
#     full_model_score_test: float
#         Balanced accuraccy of the classifier on the test set
#     """

#     # Create rocket model
#     if model_type == "rocket":
#         rocket_model = Rocket(num_kernels=num_kernels)
#     elif model_type == "minirocket":
#         rocket_model = MiniRocket(num_kernels=num_kernels)
#     elif model_type == "multirocket":
#         rocket_model = MultiRocket(num_kernels=num_kernels)
#     else:
#         raise ValueError('model_type argument should be one of the following options: (rocekt,minirocket,multirocket)')

#     # Fit rocket model
#     rocket_model.fit(X_train)
#     X_train_transform = rocket_model.transform(X_train)

#     # Scale
#     scaler = StandardScaler(with_mean=True)
#     X_train_scaled_transform = scaler.fit_transform(X_train_transform)

#     # CV to find best alpha
#     cv_classifier = RidgeClassifierCV(alphas=np.logspace(-10, 10, 20))
#     cv_classifier.fit(X_train_scaled_transform, y_train)
#     full_model_score_train = cv_classifier.score(X_train_scaled_transform, y_train)
#     best_alpha_full = cv_classifier.alpha_
#     if verbose == True:
#         print('Best Alpha:', best_alpha_full)

#     # Refit with all training set
#     classifier = RidgeClassifier(alpha=best_alpha_full)
#     classifier.fit(X_train_scaled_transform, y_train)
#     full_model_score_train = classifier.score(X_train_scaled_transform, y_train)
#     if verbose == True:
#         print('Train Accuraccy:',full_model_score_train)

#     # Transform and classify testset
#     X_test_transform = rocket_model.transform(X_test)
#     X_test_scaled_transform = scaler.transform(X_test_transform)
#     full_model_score_test = classifier.score(X_test_scaled_transform, y_test)
#     if verbose == True:
#         print('Test Accuraccy:',full_model_score_test)

#     return rocket_model, classifier, X_train_scaled_transform, X_test_scaled_transform, full_model_score_train, full_model_score_test

# def feature_detachment(classifier,
#                        X_train: np.ndarray,
#                        X_test: np.ndarray,
#                        y_train: np.ndarray,
#                        y_test: np.ndarray,
#                        drop_percentage: float = 0.05,
#                        total_number_steps: int = 150,
#                        verbose=True):
#     """
#     Applies Sequential Feature Detachment (SFD) to a feature matrix.

#     Parameters
#     ----------
#     classifier: sklearn model
#         Ridge linear classifier trained on the full training dataset.
#     X_train: numpy array
#         Training features matrix, a 2d array of shape (# instances, # features).
#     X_test: numpy array
#         Test features matrix, a 2d array of shape (# instances, # features).
#     y_train: numpy array
#         Training labels
#     y_test: nparray
#         Test labels
#     drop_percentage: float
#         Proportion of features drop at each step of the detachment process
#     total_number_steps: int
#         Total number of detachment steps performed during SFD
#     verbose: bool
#         If true, prints a message at the end of each step

#     Returns
#     -------
#     percentage_vector: numpy array
#         Array with the the model size at each detachment step (proportion of the original full model)
#     score_list_train: numpy array
#         Balanced accuraccy on the training set at each detachment step
#     score_list_test: numpy array
#         Balanced accuraccy on the test set at each detachment step
#     feature_importance_matrix: numpy array
#         A 2d array of shape (# steps, # features) with the importance of the features at each detachment step
#     """

#     # Check if training set is normalized
#     mean_vector = X_train.mean(axis=0)
#     zeros_vector = np.zeros_like(mean_vector)
#     nomalized_condition = np.isclose(mean_vector, zeros_vector, atol=1e-02)
#     assert all(nomalized_condition), "The feature matrix should be normalized before training classifier."

#     # Alpha and feature importance from full model
#     aplha_value = classifier.alpha
#     feature_importance_full = np.abs(classifier.coef_)[0, :]

#     # Define percentage vector
#     keep_percentage = 1 - drop_percentage
#     powers_vector = np.arange(total_number_steps)
#     percentage_vector = np.power(keep_percentage, powers_vector)

#     # Define lists and matrices
#     score_list_train = []
#     score_list_test = []
#     feature_importance = np.copy(feature_importance_full)
#     feature_importance_matrix = np.zeros((len(percentage_vector), len(feature_importance_full)))
#     feature_selection_matrix = np.full((len(percentage_vector), len(feature_importance_full)), False)

#     # Begin iterative feature selection
#     for count, per in enumerate(percentage_vector):

#         # Cumpute mask for selected features
#         drop_percentage = 1 - per
#         limit_value = np.quantile(feature_importance, drop_percentage)
#         selection_mask = feature_importance >= limit_value

#         # Apply mask
#         X_train_subsampled = X_train[:, selection_mask]
#         X_test_subsampled = X_test[:, selection_mask]

#         # Train model for selected features
#         step_classifier = RidgeClassifier(alpha=aplha_value)
#         step_classifier.fit(X_train_subsampled, y_train)

#         # Compute scores for train and test sets
#         # TODO add option to compute the non-balanced accuracy
#         avg_score_train = step_classifier.score(X_train_subsampled, y_train)
#         avg_score_test = step_classifier.score(X_test_subsampled, y_test)
#         score_list_train.append(avg_score_train)
#         score_list_test.append(avg_score_test)

#         # Save current feature importance and selected features
#         feature_importance_matrix[count, :] = feature_importance
#         feature_selection_matrix[count, :] = selection_mask

#         # Kill masked features
#         feature_importance[~selection_mask] = 0
#         feature_importance[selection_mask] = np.abs(step_classifier.coef_)[0, :]

#         if verbose:
#             print("Step {} out of {}".format(count + 1, total_number_steps))
#             print('{:.3f}% of features used'.format(100 * per))

#     return percentage_vector, np.asarray(score_list_train), np.asarray(
#         score_list_test), feature_importance_matrix, feature_selection_matrix


def select_optimal_model(percentage_vector,
                         acc_test,
                         full_model_score_test,
                         acc_size_tradeoff_coef: float = 0.1,
                         smoothing_points: int = 3,
                         graphics=True):
    """
    Function that selects the optimal model size after SFD process.

    Parameters
    ----------
    percentage_vector: numpy array
        Array with the the model size at each detachment step (proportion of the initial full model)
    acc_test: numpy array
        Balanced accuraccy on the test set at each detachment step
    full_model_score_test: float
        Balanced accuraccy of the full initial full classifier on the test set
    acc_size_tradeoff_coef: float
        Parameter that governs the tradeoff between size and accuraccy. 0 means only weighting accuraccy, +inf only weighting model size
    smoothing_points: int
        Level of smoothing applied to the acc_test
    graphics: bool
        If true, prints a matplotlib figure with the desition criteria

    Returns
    -------
    max_index: int
        Index of the optimal model (optimal number of SFD steps)
    max_percentage: float
        Size of the selected optimal model (proportion of the initial full model)
    """

    # Create model percentage vector
    x_vec = (1 - percentage_vector)

    # Create smoothed relative test acc vector
    y_vec = (acc_test / full_model_score_test)
    box = np.ones(smoothing_points) / smoothing_points
    y_vec_smooth = np.convolve(y_vec, box, mode='same')

    # Define the functio to optimize
    optimality_curve = acc_size_tradeoff_coef * x_vec + y_vec_smooth

    # Compute max of the function
    max_index = np.argmax(optimality_curve)
    max_x_vec = x_vec[max_index]
    max_percentage = percentage_vector[max_index]

    # Plot results
    if graphics == True:
        margin = int((smoothing_points) / 2)
        plt.plot(x_vec, y_vec, label='Relative test accuracy')
        plt.plot(x_vec[margin:-margin], optimality_curve[margin:-margin], label='Function to optimize')
        plt.scatter(max_x_vec, optimality_curve[max_index], c='C2', label='Maximum')
        plt.scatter(max_x_vec, y_vec[max_index], c='C3', label='Selected value')
        plt.legend()
        plt.ylabel('Relative Classification Accuracy')
        plt.xlabel('% of features Dropped')
        plt.show()

    return max_index, max_percentage


def retrain_optimal_model(feature_importance_matrix,
                          X_train_scaled_transform,
                          X_test_scaled_transform,
                          y_train,
                          y_test,
                          full_model_score_train,
                          full_model_score_test,
                          original_best_alpha,
                          max_index,
                          verbose=True):
    """
    Function that retrains a classifier with the optimal subset of selected features.

    Parameters
    ----------
    feature_importance_matrix: numpy array
        A 2d array of shape (# steps, # features) with the importance of the features at each detachment step
    X_train_scaled_transform: numpy array
        Training features matrix, a 2d array of shape (# instances, # dimensions)
    X_test_scaled_transform: numpy array
        Test features matrix, a 2d array of shape (# instances, # dimensions)
    full_model_score_train: float
        Balanced accuraccy of the full initial full classifier on the train set
    full_model_score_test: float
        Balanced accuraccy of the full initial full classifier on the test set
    original_best_alpha: float
        Alpha regularization parameter of the full initial ridge classifier
    max_index: int
        Index of the optimal model (optimal number of SFD steps)
    verbose: bool
        If true, prints the results

new_alpha_classifier, new_alpha_acc_train, new_alpha_acc_test, old_alpha_classifier, old_alpha_acc_train, old_alpha_acc_test

    Returns
    -------
    new_alpha_classifier: sklearn model
        Ridge classifier trained on selected features, alpha is recomputed from the training set
    new_alpha_acc_train: float
        Balanced accuracy on the training set with new_alpha_classifier
    new_alpha_acc_train: float
        Balanced accuracy on the test set with new_alpha_classifier
    old_alpha_classifier: sklearn model
        Ridge classifier trained on selected features with alpha value equal to the full model
    old_alpha_acc_train: float
        Balanced accuracy on the training set with old_alpha_acc_train
    old_alpha_acc_train: float
        Balanced accuracy on the test set with old_alpha_acc_train
    """

    feature_mask = feature_importance_matrix[max_index] > 0
    masked_X_train = X_train_scaled_transform[:, feature_mask]
    masked_X_test = X_test_scaled_transform[:, feature_mask]

    # REFIT WITH NEW ALPHA

    # CV to find best alpha
    cv_classifier = RidgeClassifierCV(alphas=np.logspace(-10, 10, 20))
    cv_classifier.fit(masked_X_train, y_train)
    best_alpha_full = cv_classifier.alpha_

    # Refit with all training set
    new_alpha_classifier = RidgeClassifier(alpha=best_alpha_full)
    new_alpha_classifier.fit(masked_X_train, y_train)
    new_alpha_acc_train = new_alpha_classifier.score(masked_X_train, y_train)

    # Transform and classify testset
    new_alpha_classifier.score(masked_X_test, y_test)
    new_alpha_acc_test = new_alpha_classifier.score(masked_X_test, y_test)

    # REFIT WITH ORIGINA ALPHA

    # Refit with all training set
    old_alpha_classifier = RidgeClassifier(alpha=original_best_alpha)
    old_alpha_classifier.fit(masked_X_train, y_train)
    old_alpha_acc_train = old_alpha_classifier.score(masked_X_train, y_train)

    # Transform and classify testset
    old_alpha_classifier.score(masked_X_test, y_test)
    old_alpha_acc_test = old_alpha_classifier.score(masked_X_test, y_test)

    if verbose == True:
        print('Full model train ACC: {:.2f}%'.format(100 * full_model_score_train))
        print('Full model test ACC: {:.2f}%'.format(100 * full_model_score_test))
        print(' ')
        print('------------')
        print(' ')
        print('RESULTS WITH NEW ALPHA RESULTS:')
        print('New Best Alpha: ', best_alpha_full)
        print(' ')
        print('Train Accuraccy: {:.2f}%'.format(100 * new_alpha_acc_train))
        print('Test Accuraccy: {:.2f}%'.format(100 * new_alpha_acc_test))
        print(' ')
        print('------------')
        print(' ')
        print('RESULTS WITH ORIGINAL ALPHA RESULTS:')
        print('Original Best Alpha:', original_best_alpha)
        print(' ')
        print('Train Accuraccy: {:.2f}%'.format(100 * old_alpha_acc_train))
        print('Test Accuraccy: {:.2f}%'.format(100 * old_alpha_acc_test))

    return new_alpha_classifier, new_alpha_acc_train, new_alpha_acc_test, old_alpha_classifier, old_alpha_acc_train, old_alpha_acc_test


def compute_power_frequency(vector, sampling_rate, num_bins):
    # Compute the power spectrum using Welch's method
    f, Pxx = welch(vector, fs=sampling_rate)
    
    # # Calculate bin width based on the number of bins
    # bin_width = f[-1] / num_bins
    
    # binned_power = np.zeros(num_bins)
    # center_frequencies = np.zeros(num_bins)
    
    # for i in range(num_bins):
    #     freq_range = [i * bin_width, (i + 1) * bin_width]
    #     indices = np.where((f >= freq_range[0]) & (f < freq_range[1]))
    #     binned_power[i] = np.mean(Pxx[indices])
    #     center_frequencies[i] = np.mean(f[indices])
    
    # # Normalize the binned power spectrum
    # normalized_power = binned_power / np.sum(binned_power)
    # return normalized_power, center_frequencies

    normalized_power = Pxx / np.sum(Pxx)

    return normalized_power, f

def compute_average_power_spectrum(time_series_list, sampling_rate, num_bins):

    power_spectrum, center_frequencies = compute_power_frequency(time_series_list[0], sampling_rate, num_bins)
    total_power_spectrum = np.zeros(len(center_frequencies))
    
    for time_series in time_series_list:
        power_spectrum, center_frequencies = compute_power_frequency(time_series, sampling_rate, num_bins)
        total_power_spectrum += power_spectrum
    
    average_power_spectrum = total_power_spectrum / len(time_series_list)
    # print('here')

    return average_power_spectrum, center_frequencies

def plot_power_spectrum(center_frequencies, power_spectrum):
    plt.figure(figsize=(10, 6))
    plt.bar(center_frequencies, power_spectrum, width=center_frequencies[1] - center_frequencies[0], align='center')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Power')
    plt.title('Power Spectrum')
    plt.grid(True)
    plt.show()



def feature_detachment(classifier,
                        X_train: np.ndarray,
                        X_test: np.ndarray,
                        y_train: np.ndarray,
                        y_test: np.ndarray,
                        drop_percentage: float = 0.05,
                        total_number_steps: int = 150,
                        multilabel_type: str = "norm",
                        verbose = True):
    """
    Applies Sequential Feature Detachment (SFD) to a feature matrix.

    Parameters
    ----------
    classifier: sklearn model
        Ridge linear classifier trained on the full training dataset.
    X_train: numpy array
        Training features matrix, a 2d array of shape (# instances, # features).
    X_test: numpy array
        Test features matrix, a 2d array of shape (# instances, # features).
    y_train: numpy array
        Training labels
    y_test: nparray
        Test labels
    drop_percentage: float
        Proportion of features drop at each step of the detachment process
    total_number_steps: int
        Total number of detachment steps performed during SFD
    verbose: bool
        If true, prints a message at the end of each step

    Returns
    -------
    percentage_vector: numpy array
        Array with the the model size at each detachment step (proportion of the original full model)
    score_list_train: numpy array
        Balanced accuraccy on the training set at each detachment step
    score_list_test: numpy array
        Balanced accuraccy on the test set at each detachment step
    feature_importance_matrix: numpy array
        A 2d array of shape (# steps, # features) with the importance of the features at each detachment step
    """

    # Check if training set is normalized
    mean_vector = X_train.mean(axis=0)
    zeros_vector = np.zeros_like(mean_vector)
    nomalized_condition = np.isclose(mean_vector, zeros_vector,atol=1e-02)
    # assert all(nomalized_condition), "The feature matrix should be normalized before training classifier."
    total_feats = X_train.shape[1]

    # Feature importance from full model
    
    # Check if problem is multilabel
    if np.shape(classifier.coef_)[0]>1:
        if multilabel_type == "norm":
            feature_importance_full = np.linalg.norm(classifier.coef_[:,:],axis=0,ord=2)
        elif multilabel_type == "max":
            feature_importance_full = np.linalg.norm(classifier.coef_[:,:],axis=0,ord=np.inf)
        elif multilabel_type == "avg":
            feature_importance_full = np.linalg.norm(classifier.coef_[:,:],axis=0,ord=1)
        else:
            raise ValueError('Invalid multilabel_type argument. Choose from: "norm", "max", or "avg".')
    else:
        feature_importance_full = np.abs(classifier.coef_)[0,:]

    # Define percentage vector
    keep_percentage = 1-drop_percentage
    powers_vector = np.arange(total_number_steps)
    percentage_vector_unif = np.power(keep_percentage, powers_vector)

    num_feat_per_step = np.unique((percentage_vector_unif*total_feats).astype(int))
    num_feat_per_step = num_feat_per_step[::-1]
    num_feat_per_step = num_feat_per_step[num_feat_per_step>0]
    percentage_vector = num_feat_per_step/total_feats

    # Define lists and matrices
    score_list_train = []
    score_list_test = []
    feature_importance = np.copy(feature_importance_full)
    feature_importance_matrix = np.zeros((len(percentage_vector),len(feature_importance_full)))
    feature_selection_matrix = np.full((len(percentage_vector),len(feature_importance_full)), False)

    # Begin iterative feature selection
    for count, feat_num in enumerate(num_feat_per_step):

        per = percentage_vector[count]

        # Cumpute mask for selected features
        drop_features = total_feats - feat_num

        selected_idxs = np.argsort(feature_importance)[drop_features:]
        selection_mask = np.full(total_feats, False)
        selection_mask[selected_idxs] = True

        # Apply mask
        X_train_subsampled = X_train[:,selection_mask]
        X_test_subsampled = X_test[:,selection_mask]

        # Train model for selected features
        classifier.fit(X_train_subsampled, y_train)

        # Compute scores for train and test sets
        avg_score_train = classifier.score(X_train_subsampled, y_train)
        avg_score_test = classifier.score(X_test_subsampled, y_test)
        score_list_train.append(avg_score_train)
        score_list_test.append(avg_score_test)

        # Kill masked features
        feature_importance[~selection_mask] = 0

        # Compute feature importance taking into account multilabel type
        if np.shape(classifier.coef_)[0]>1:
            if multilabel_type == "norm":
                feature_importance[selection_mask] = np.linalg.norm(classifier.coef_[:,:],axis=0,ord=2)
            elif multilabel_type == "max":
                feature_importance[selection_mask] = np.linalg.norm(classifier.coef_[:,:],axis=0,ord=np.inf)
            elif multilabel_type == "avg":
                feature_importance[selection_mask] = np.linalg.norm(classifier.coef_[:,:],axis=0,ord=1)
            else:
                raise ValueError('Invalid multilabel_type argument. Choose from: "norm", "max", or "avg".')
        else:
            feature_importance[selection_mask] = np.abs(classifier.coef_)[0,:]

        if verbose==True:
            print("Step {} out of {}".format(count+1, total_number_steps))
            print('{:.3f}% of features used'.format(100*per))

        # Save current feature importance and selected features
        feature_importance_matrix[count,:] = feature_importance
        feature_selection_matrix[count,:] = selection_mask
        # print('Im here')

    return percentage_vector, np.asarray(score_list_train), np.asarray(score_list_test), feature_importance_matrix, feature_selection_matrix