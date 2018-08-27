import numpy as np 
import math
import bastien_utils
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn import svm

def reconstructions(atoms, encodings):
    """
    Return the images reconstructed by taking the linear combination of the atoms with the various weights of the encoding.
        atoms: shape (nb_atoms, nb_rows, nb_columns, nb_channels)
        encodings: shape (nb_samples, nb_atoms)
    """
    nb_atoms, nb_rows, nb_columns, nb_channels = atoms.shape
    nb_samples, _ = encodings.shape
    x_rec = np.dot(encodings, atoms.reshape(nb_atoms, nb_rows*nb_columns*nb_channels)).reshape(nb_samples, nb_rows, nb_columns, nb_channels)
    return x_rec
    
def plot_reconstructions(atoms, encodings):
    """
    Plots the images reconstructed by taking the linear combination of the atoms with the various weights of the encoding.
        atoms: shape (nb_atoms, nb_rows, nb_columns, nb_channels)
        encodings: shape (nb_samples, nb_atoms)
    """
    x_rec = reconstructions(atoms, encodings)
    bastien_utils.plot_all_images(x_rec)

def reconstruction_error(x_original, atoms, encodings):
    """
    Return the mse error between the original images and the reconstructions.
    x_original: shape (nb_samples, nb_rows, nb_columns, nb_channels) 
    atoms: shape (nb_atoms, nb_rows, nb_columns, nb_channels)
    encodings: shape (nb_samples, nb_atoms)
    """
    x_rec = reconstructions(atoms, encodings)
    return np.mean(np.square(x_original-x_rec))

def sparsity_Hoyer(encodings):
    """
    Returns the mean of the sparsity measure of the set of encodings as defined in Hoyer 2004.
    The measure is equal to 0 iif all entries of a vector are equals, and 1 iif all but one are equal to zero.
    encodings: shape (nb_samples, nb_dims)
    """
    latent_dim = encodings.shape[1]
    sqrt = math.sqrt(latent_dim)
    sigma = (sqrt - (np.linalg.norm(encodings, ord=1, axis=1)/(np.linalg.norm(encodings, ord=2, axis=1)+0.0000001)))/(sqrt - 1)
    return np.mean(sigma)

def max_approximation(atoms, encodings, operator, **kwargs):
    """
    Returns the max-approximation to the operator, as defined in [Angulo, Velasco-Forerro 2017], by applying the operator to all the atoms.
        atoms: (nb_atoms, nb_rows, nb_columns, nb_channels) numpy array
        encodings: (nb_samples, nb_atoms)
        operator: function that must have at least one argument: a (nb_samples, nb_rows, nb_columns, nb_channels) numpy array
        **kwargs: dictionnary of keyword arguments of the operator 
    """
    nb_atoms, nb_rows, nb_columns, nb_channels = atoms.shape
    nb_samples, _ = encodings.shape
    dilated_atoms = bastien_utils.apply_operator_to_all_images(operator, atoms, **kwargs)
    max_approx = np.dot(encodings,dilated_atoms.reshape(nb_atoms, nb_rows*nb_columns*nb_channels)).reshape(nb_samples, nb_rows, nb_columns, nb_channels)
    return max_approx

def plot_max_approximation(atoms, encodings, operator, **kwargs):
    """
    Plots the max-approximation to the operator, as defined in [Angulo, Velasco-Forerro 2017], by applying the operator to all the atoms.
        atoms: (nb_atoms, nb_rows, nb_columns, nb_channels) numpy array
        encodings: (nb_samples, nb_atoms)
        operator: function that must have at least one argument: a (nb_samples, nb_rows, nb_columns, nb_channels) numpy array
        **kwargs: dictionnary of keyword arguments of the operator 
    """
    max_approx = max_approximation(atoms, encodings, operator, **kwargs)
    bastien_utils.plot_all_images(max_approx)

def max_approximation_error(x_original, atoms, encodings, operator, **kwargs):
    """
    Returns the mse errors between the max-approximation to the operator, as defined in [Angulo, Velasco-Forerro 2017]
        and the operator applied to the original images (first returned value) and to the reconstructions (second returned value)
    Arguments:
        x_original: (nb_samples, nb_rows, nb_columns, nb_channels) original images
        atoms: (nb_atoms, nb_rows, nb_columns, nb_channels) atoms of the dictionary
        encodings: (nb_samples, nb_atoms) the encoding the original images
        opertor: a callable object that takes as first argument an image as a numpy array (nb_rows, nb_columns)
    """
    x_dilatation = bastien_utils.apply_operator_to_all_images(operator, x_original, **kwargs)
    x_rec_dilated = bastien_utils.apply_operator_to_all_images(operator, reconstructions(atoms, encodings), **kwargs) 
    dilated_atoms = bastien_utils.apply_operator_to_all_images(operator, atoms, **kwargs)
    max_appox_err_to_original = reconstruction_error(x_dilatation, dilated_atoms, encodings)
    max_appox_err_to_rec = reconstruction_error(x_rec_dilated, dilated_atoms, encodings)
    return max_appox_err_to_original, max_appox_err_to_rec

def best_SVM_classification_score(encodings, labels, min_log_C=-2, max_log_C=3, nb_values_C=10, min_log_gamma=-3, max_log_gamma=2, nb_values_gamma=10):
    """
    Performs cross-validation and parameter seletion (grid search on the specified parameter) 
        and returns the best classification score using a SVM classifier with a gaussian kernel (rbf).
    Returns also the best parameters on the grid search, as a dictionary.
    Arguments:
        encodings: numpy array (nb_samples, nb_dims) encoding of the images.
        labels: numpy array (nb_samples,). Labels of each image for the classification.
        min_log_C: int. Minimal value of the C parameter of the SVM classifier (margin) to be tested in log_10 scale.
        max_log_C: int.  Maximal value of the C parameter of the SVM classifier (margin) to be tested in log_10 scale.
        nb_values_C: int. Number of values of the C parameter of the SVM classifier (margin) to be tested in log_10 scale.
        min_log_gamma: int. Minimal value of the gamma parameter of the SVM classifier (free parameter) to be tested in log_10 scale.
        max_log_gamma: int.  Maximal value of the C parameter of the SVM classifier (free paramter) to be tested in log_10 scale.
        nb_values_gamma: int. Number of values of the C parameter of the SVM classifier (free paramter) to be tested in log_10 scale.
    Note that the larger gamma is, the smaller the distance between two points must be for the kernel value to be close to 1.
    A value larger than 100 (max_log_C=2) usually leads to poor classification performance.
    On the opposite the smaller gamma, the more points are considered in the 'neighborhood' of each specific point.
    """
    H_train, H_test, Y_train, Y_test = train_test_split(encodings, labels, test_size=0.1)
    C_range = np.logspace(min_log_C, max_log_C, nb_values_C)
    gamma_range = np.logspace(min_log_gamma, max_log_gamma, nb_values_gamma)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(cache_size=600), param_grid=param_grid, cv=cv, verbose=2)
    grid.fit(H_train, Y_train)
    return grid.score(H_test, Y_test), grid.best_params_ 