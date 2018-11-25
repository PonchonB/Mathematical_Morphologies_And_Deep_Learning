import numpy as np 
import math
import bastien_utils
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn import svm
from matplotlib import pyplot as plt
from skimage import filters


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

def plot_histograms_of_the_encoding(H):
    """
    Plots the histogram of each of the 10 first rows of H (nb_samples, nb_features), that is the encoding of each of the nb_samples images
    """
    nb_samples, nb_features = H.shape
    plt.figure(figsize=(30, 4))
    for i in range(10):
        ax = plt.subplot(1, 10, i + 1)
        ax.hist(H[i], bins=nb_features)
    plt.show()    

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

def best_linearSVM_classification_score(encodings, labels, min_log_C=-2, max_log_C=3, nb_values_C=10):
    """
    Performs cross-validation and parameter seletion (grid search on the specified parameter) 
        and returns the best classification score using a SVM classifier with a Linear kernel, assumed to be sufficient when the number of features is large.
    Returns also the best C on the grid search, as a dictionary (with only one element).
    Arguments:
        X: numpy array (nb_samples, nb_rows, nb_columns, nb_input_channels) to be encoded on whose code will be use for the classification
        y: numpy array (nb_sampls,). Labels of each image for the classification.
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
    H_train, H_test, Y_train, Y_test = train_test_split(encodings, labels, test_size=0.2, stratify=labels)
    C_range = np.logspace(min_log_C, max_log_C, nb_values_C)
    param_grid = dict(C=C_range)
    cv = StratifiedShuffleSplit(n_splits=20, test_size=0.08, random_state=42)
    grid = GridSearchCV(svm.LinearSVC(dual=False), param_grid=param_grid, cv=cv, verbose=2, refit=False)
    grid.fit(H_train, Y_train)
    scores = np.zeros(25)
    cv = StratifiedShuffleSplit(n_splits=25, test_size=0.05)
    i=0
    for train_index, test_index in cv.split(H_test, Y_test):
        H_test_train = H_test[train_index]
        Y_test_train = Y_test[train_index]
        H_test_test = H_test[test_index]
        Y_test_test = Y_test[test_index]
        linearSVM = svm.LinearSVC(dual=False, **grid.best_params_)
        linearSVM.fit(H_test_train, Y_test_train)
        scores[i] = linearSVM.score(H_test_test, Y_test_test)
        i=i+1
    return np.mean(scores), np.std(scores), grid.best_params_


def sparsity_KL_divergence(encodings, sparsity_objective=0.01, sparsity_weight=1, multiply_by_weight_penalty=True):
    """
    Computes the KL divergence sparsity measure of the encodings, with a specific set of parameters of the cost function.
        sparsity_objective: float in [0,1].
        sparsity_weight: positive float.
        multiply_by_weight_penalty: bool. Weather to multyply the loss function by a weighting term (the sparsity_weight parameter).
    """
    s_hat = np.mean(encodings, axis=0)
    np.clip(s_hat, 0.0000001, 1)
    val = sparsity_objective*np.log(sparsity_objective/s_hat) + (1-sparsity_objective)*np.log((1-sparsity_objective)/(1-s_hat))
    if not multiply_by_weight_penalty:
        sparsity_weight=1
    return sparsity_weight*np.sum(val)

def mean_Otsu_binarization_threshold_using_majors_images(images, h_test_image, nb_images_to_use=10):
    nb_images,_,_ = images.shape
    thresholds = np.zeros(nb_images_to_use)
    idx_sort = np.flip(np.argsort(h_test_image))
    for i in range(nb_images_to_use):
        thresholds[i] = filters.threshold_otsu(images[idx_sort[i]])
    return np.mean(thresholds)

def overlap_coef(A,B):
    if (not np.any(A)) or (not np.any(B)):
        #print("Warning: Two empty sets given in computation of jaccard index...returning 0...")
        return 0
    else:
        A_union_B = np.logical_or(A,B)
        A_inter_B = np.logical_and(A,B)
        overlap_coef = np.sum(A_inter_B)/min(np.sum(A), np.sum(B))
        return overlap_coef

def pair_wise_mean_overlap_coef_of_binary_images(binary_images):
    nb_images,_,_ = binary_images.shape
    nb_pairs = nb_images*(nb_images-1)/2.
    s = 0
    for i in range(nb_images):
        for j in range(i+1,nb_images):
            s += overlap_coef(binary_images[i], binary_images[j])
    return s/nb_pairs

def pair_wise_mean_overlap_coef_of_weighted_atoms(atoms, h_test_image, nb_images_to_use_for_threshold_computation=10):
    nb_atoms,nb_rows,nb_columns,_ = atoms.shape
    weighted_atoms = np.dot(np.diag(h_test_image), atoms.reshape((nb_atoms,nb_rows*nb_columns))).reshape((nb_atoms,nb_rows,nb_columns))
    binarization_theshold = mean_Otsu_binarization_threshold_using_majors_images(weighted_atoms, h_test_image, nb_images_to_use=nb_images_to_use_for_threshold_computation)
    print('*****Otsu binarization  threshold : ', binarization_theshold,' ******')
    bin_weighted_atoms = weighted_atoms > binarization_theshold
    return pair_wise_mean_overlap_coef_of_binary_images(bin_weighted_atoms)

def mean_overlap_coef_of_atoms_weighted_by_images_code(atoms, h_test, nb_images_to_use_for_threshold_computation=10):
    nb_samples, _ = h_test.shape
    s = 0
    for i in range(nb_samples):
        print('******Image: ', i, ' ******')
        tmp = pair_wise_mean_overlap_coef_of_weighted_atoms(atoms, h_test[i], nb_images_to_use_for_threshold_computation=nb_images_to_use_for_threshold_computation)
        s += tmp
        print('***Mean overlap coef: ', tmp)
        print('\n')
    return s/nb_samples

def jaccard_index(A,B):
    if (not np.any(A)) and (not np.any(B)):
        #print("Warning: Two empty sets given in computation of jaccard index...returning 0...")
        return 0
    else:
        A_union_B = np.logical_or(A,B)
        A_inter_B = np.logical_and(A,B)
        jaccard_index = np.sum(A_inter_B)/np.sum(A_union_B)
        return jaccard_index

def pair_wise_mean_Jaccard_index_of_binary_images(binary_images):
    nb_images,_,_ = binary_images.shape
    nb_pairs = nb_images*(nb_images-1)/2.
    s = 0
    for i in range(nb_images):
        for j in range(i+1,nb_images):
            s += jaccard_index(binary_images[i], binary_images[j])
    return s/nb_pairs

def pair_wise_mean_Jaccard_index_of_weighted_atoms(atoms, h_test_image, nb_images_to_use_for_threshold_computation=10):
    nb_atoms,nb_rows,nb_columns,_ = atoms.shape
    weighted_atoms = np.dot(np.diag(h_test_image), atoms.reshape((nb_atoms,nb_rows*nb_columns))).reshape((nb_atoms,nb_rows,nb_columns))
    binarization_theshold = mean_Otsu_binarization_threshold_using_majors_images(weighted_atoms, h_test_image, nb_images_to_use=nb_images_to_use_for_threshold_computation)
    print('*****Otsu binarization  threshold : ', binarization_theshold,' ******')
    bin_weighted_atoms = weighted_atoms > binarization_theshold
    return pair_wise_mean_Jaccard_index_of_binary_images(bin_weighted_atoms)

def mean_jaccard_index_of_atoms_weighted_by_images_code(atoms, h_test, nb_images_to_use_for_threshold_computation=10):
    nb_samples, _ = h_test.shape
    s = 0
    for i in range(nb_samples):
        print('******Image: ', i, ' ******')
        tmp = pair_wise_mean_Jaccard_index_of_weighted_atoms(atoms, h_test[i], nb_images_to_use_for_threshold_computation=nb_images_to_use_for_threshold_computation)
        s += tmp
        print('***Mean Jaccard Index: ', tmp)
        print('\n')
    return s/nb_samples