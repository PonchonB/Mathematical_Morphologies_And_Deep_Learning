import numpy as np
import bastien_utils
from matplotlib import pyplot as plt
from skimage import filters
import math

def idx_to_sort_code_coefs_by_decreasing_value(h_image):
    return np.flip(np.argsort(h_image))


def k_highest_code_coefs_and_associated_atoms(k, h_image, atoms):
    idx_sort = idx_to_sort_code_coefs_by_decreasing_value(h_image)
    major_atoms = atoms[idx_sort[:k]]
    major_h = h_image[idx_sort[:k]]
    return major_h, major_atoms

def k_most_used_atoms(k, h_image, atoms):
    _, major_atoms = k_highest_code_coefs_and_associated_atoms(k,h_image,atoms)
    return major_atoms

def plot_10_most_used_atoms_for_an_image(h_image, atoms, same_intensity_scale = True):
    """
    Arguments:
        h_image: numpy array of shape (N_features,), the encoding of an image
        atoms: numpy array of shape (N_features, N_pixels), the atom images of the learned representation
    """
    major_atoms=k_most_used_atoms(10,h_image,atoms)
    print('Atoms associated with the 10 highest code coefficients of the image')
    bastien_utils.plot_all_images(major_atoms, same_intensity_scale=same_intensity_scale)

def plot_most_used_atom_for_each_image(h_images, atoms):
    """
    Arguments:
        h_images: numpy array of shape (Nb_images,Nb_features), the encoding of the images
        atoms: numpy array of shape (Nb_features, Nb_pixels), the atom images of the learned representation
    """
    nb_images = min(100, h_images.shape[0])
    _, nb_rows, nb_columns, _ = atoms.shape 
    most_used_atoms = np.zeros((nb_images, nb_rows,nb_columns, 1))
    for i in range(nb_images):
        most_used_atoms[i] = k_most_used_atoms(1, h_images[i], atoms)
    print("Most used atom for each image.")
    bastien_utils.plot_all_images(most_used_atoms, same_intensity_scale=True)

def progressive_reconstruction(h_image, atoms, number_of_steps=50):
    """
    Plot number_of_steps images, each showing the reconstruction of the image whose encoding is given by h_image, 
        using only the k atom associated with the highest code coeffciients. 
    Argument:
        h_image: numpy array of shape (Nb_features,), the encoding of the image.
        atoms: numpy array of shape (Nb_features, Nb_pixels), the atom images of the learned representation.
        number_of_steps: int, the maximum number of atoms to use in the reconstructions (and hence the number of images showed)
    """
    _, nb_rows, nb_columns, _ = atoms.shape
    h_sort, major_atoms = k_highest_code_coefs_and_associated_atoms(number_of_steps, h_image, atoms)
    major_atoms = major_atoms.reshape((number_of_steps, nb_rows*nb_columns))
    partial_reconstructions = np.zeros((number_of_steps, nb_rows*nb_columns))
    for i in range(number_of_steps):
        partial_reconstructions[i,:] = np.dot(h_sort[:i+1], major_atoms[:i+1])
    bastien_utils.plot_all_images(partial_reconstructions.reshape(number_of_steps, nb_rows, nb_columns, 1), same_intensity_scale=True)
    
def plot_reconstruction_using_three_most_used_atoms_as_rgb(h_image, atoms):
    """
    Arguments:
        h_image: numpy array of shape (Nb_features,), the encoding of the image.
        atoms: numpy array of shape (Nb_features, Nb_pixels), the atom images of the learned representation.
    """
    _, nb_rows, nb_columns, _ = atoms.shape
    major_h, major_atoms = k_highest_code_coefs_and_associated_atoms(3, h_image, atoms)
    major_atoms = major_atoms.reshape((3, nb_rows*nb_columns))
    rgb_image = np.swapaxes(np.dot(np.diag(major_h), major_atoms), 0, 1).reshape((nb_rows,nb_columns,3))
    vmax=np.max(rgb_image)
    rgb_image = rgb_image/vmax
    plt.imshow(rgb_image)
    plt.show()

def atoms_weighted_by_encoding_coefficients(atoms, h_test_image):
    nb_atoms,nb_rows,nb_columns,_ = atoms.shape
    weighted_atoms = np.dot(np.diag(h_test_image), atoms.reshape((nb_atoms,nb_rows*nb_columns))).reshape((nb_atoms,nb_rows,nb_columns, 1))
    return weighted_atoms

def plot_weighted_atoms(atoms, h_test_image):
    weighted_atoms = atoms_weighted_by_encoding_coefficients(atoms, h_test_image)
    bastien_utils.plot_all_images(weighted_atoms, same_intensity_scale=True)

def mean_Otsu_binarization_threshold_using_majors_atoms(atoms, h_test_image, nb_atoms_to_use=10):
    """
    Returns the mean of the Otsu binarization thresholds of the atoms associated with the nb_atoms_to_use highest code coefficients.
    Arguments:
        atoms: numpy array of shape (Nb_features, Nb_pixels), the atom images of the learned representation.
        h_image: numpy array of shape (Nb_features,), the encoding of the image.
        nb_atoms_to_use: int, the number of atoms on which the Otsu binarization threshold wil be computed.
    """
    thresholds = np.zeros(nb_atoms_to_use)
    idx_sort = idx_to_sort_code_coefs_by_decreasing_value(h_test_image)
    for i in range(nb_atoms_to_use):
        thresholds[i] = filters.threshold_otsu(atoms[idx_sort[i]])
    return np.mean(thresholds)

def binarized_weighted_atoms(atoms, h_test_image, nb_atoms_to_use_for_threshold_computation=20):
    weighted_atoms = atoms_weighted_by_encoding_coefficients(atoms, h_test_image)
    binarization_theshold = mean_Otsu_binarization_threshold_using_majors_atoms(weighted_atoms, h_test_image, nb_atoms_to_use=nb_atoms_to_use_for_threshold_computation)
    # print('*****Otsu binarization  threshold : ', binarization_theshold,' ******')
    bin_weighted_atoms = weighted_atoms > binarization_theshold
    return bin_weighted_atoms

def plot_binarized_weighted_atoms(atoms, h_test_image, nb_atoms_to_use_for_threshold_computation=20):
    bin_weighted_atoms = binarized_weighted_atoms(atoms, h_test_image, nb_atoms_to_use_for_threshold_computation=nb_atoms_to_use_for_threshold_computation) 
    bastien_utils.plot_all_images(bin_weighted_atoms, same_intensity_scale=True)

def overlap_coef(A,B):
    if (not np.any(A)) or (not np.any(B)):
        #print("Warning: Two empty sets given in computation of jaccard index...returning 0...")
        return 0
    else:
        A_inter_B = np.logical_and(A,B)
        overlap_coef = np.sum(A_inter_B)/min(np.sum(A), np.sum(B))
        return overlap_coef

def jaccard_index(A,B):
    if (not np.any(A)) and (not np.any(B)):
        #print("Warning: Two empty sets given in computation of jaccard index...returning 0...")
        return 0
    else:
        A_union_B = np.logical_or(A,B)
        A_inter_B = np.logical_and(A,B)
        jaccard_index = np.sum(A_inter_B)/np.sum(A_union_B)
        return jaccard_index

def gray_scale_jaccard_index(image1, image2):
    assert (np.all(image1>=0) and np.all(image2>=0))
    if (np.all(image1==0) and np.all(image2==0)):
        return 0
    else:
        min_image = np.minimum(image1, image2)
        max_image = np.maximum(image1, image2)
        return np.sum(min_image)/float(np.sum(max_image))

def mean_of_function_on_all_pairs_of_images(function, images): 
    nb_images,_,_ = images.shape
    nb_pairs = nb_images*(nb_images-1)/2.
    s = 0
    for i in range(nb_images):
        for j in range(i+1,nb_images):
            s += function(images[i], images[j])
    return s/nb_pairs

# def mean_of_overlapcoef_on_all_pairs_of_images(binary_images):
#     return mean_of_function_on_all_pairs_of_images(overlap_coef, binary_images)

# def mean_of_Jaccard_index_on_all_pairs_of_images(binary_images):
#     return mean_of_function_on_all_pairs_of_images(jaccard_index, binary_images)

# def mean_of_gray_scale_jaccard_index_on_all_pairs_of_images(images):
#     return mean_of_function_on_all_pairs_of_images(gray_scale_jaccard_index, images)

def mean_of_function_on_all_pairs_of_weighted_atoms(function, atoms, h_test_image, binarize=True, nb_atoms_to_use_for_threshold_computation=20):
    if binarize:
        weighted_atoms = binarized_weighted_atoms(atoms, h_test_image, nb_atoms_to_use_for_threshold_computation=nb_atoms_to_use_for_threshold_computation)[:,:,:,0]
    else:
        weighted_atoms = atoms_weighted_by_encoding_coefficients(atoms, h_test_image)[:,:,:,0]
    return mean_of_function_on_all_pairs_of_images(function, weighted_atoms)

# def pair_wise_mean_overlap_coef_of_weighted_atoms(atoms, h_test_image, nb_atoms_to_use_for_threshold_computation=20):
#     bin_weighted_atoms = binarized_weighted_atoms(atoms, h_test_image, nb_atoms_to_use_for_threshold_computation=nb_atoms_to_use_for_threshold_computation)[:,:,:,0]
#     return pair_wise_mean_overlap_coef_of_binary_images(bin_weighted_atoms)

# def pair_wise_mean_Jaccard_index_of_weighted_atoms(atoms, h_test_image, nb_atoms_to_use_for_threshold_computation=20):
#     bin_weighted_atoms = binarized_weighted_atoms(atoms, h_test_image, nb_atoms_to_use_for_threshold_computation=nb_atoms_to_use_for_threshold_computation)[:,:,:,0]
#     return pair_wise_mean_Jaccard_index_of_binary_images(bin_weighted_atoms)

# def mean_on_all_test_images_of_function_of_binarized_weighted_atoms(function, atoms, h_test, nb_atoms_to_use_for_threshold_computation=20):
#     value_for_each_image = np.apply_along_axis(
#                                 lambda x:mean_of_function_on_all_pairs_of_binarized_weighted_atoms(function, atoms, x, nb_atoms_to_use_for_threshold_computation=nb_atoms_to_use_for_threshold_computation),
#                                                                                                     1, h_test)
#     return np.mean(value_for_each_image)

def mean_on_all_test_images_of_function_of_weighted_atoms(function, atoms, h_test, binarize=True, nb_atoms_to_use_for_threshold_computation=20):
    nb_samples, _ = h_test.shape
    s = 0
    for i in range(nb_samples):
        # print('******Image: ', i, ' ******')
        s += mean_of_function_on_all_pairs_of_weighted_atoms(function, atoms, h_test[i], binarize=binarize, nb_atoms_to_use_for_threshold_computation=nb_atoms_to_use_for_threshold_computation)
        # s += tmp
        # print('***Value: ', tmp)
        # print('\n')
    return s/nb_samples

def mean_overlap_coef_of_atoms_weighted_by_images_code(atoms, h_test, nb_atoms_to_use_for_threshold_computation=20):
    return mean_on_all_test_images_of_function_of_weighted_atoms(overlap_coef, atoms, h_test, binarize=True, nb_atoms_to_use_for_threshold_computation=nb_atoms_to_use_for_threshold_computation)

def mean_jaccard_index_of_atoms_weighted_by_images_code(atoms, h_test, nb_atoms_to_use_for_threshold_computation=20):
    return mean_on_all_test_images_of_function_of_weighted_atoms(jaccard_index, atoms, h_test, binarize=True, nb_atoms_to_use_for_threshold_computation=nb_atoms_to_use_for_threshold_computation)

def mean_gray_scale_jaccard_index_of_atoms_weighted_by_images_code(atoms, h_test):
    return mean_on_all_test_images_of_function_of_weighted_atoms(gray_scale_jaccard_index, atoms, h_test, binarize=False)

def plot_heat_map_of_overlapping_atoms(h_images, atoms, dilated_atoms=None, number_of_atoms=20, nb_atoms_to_use_for_threshold_computation=20):
    nb_atoms, nb_rows, nb_columns, _ = atoms.shape
    nb_samples_to_plot = min(h_images.shape[0], 8)
    number_of_atoms = min(number_of_atoms, nb_atoms)
    if dilated_atoms is None:
        show_dilated_atoms = False
        n_rows_in_figure = 1
    else:
        show_dilated_atoms = True
        n_rows_in_figure = 2
        heat_map_of_dilated_atoms = np.zeros((nb_samples_to_plot,nb_rows, nb_columns))
    heat_map_of_atoms = np.zeros((nb_samples_to_plot,nb_rows, nb_columns))
    for i in range(nb_samples_to_plot):
        idx_sort = idx_to_sort_code_coefs_by_decreasing_value(h_images[i])
        bin_weighted_atoms = binarized_weighted_atoms(atoms, h_images[i], 
                                            nb_atoms_to_use_for_threshold_computation=nb_atoms_to_use_for_threshold_computation)
        bin_weighted_atoms_to_use = bin_weighted_atoms[idx_sort[:number_of_atoms]].reshape((number_of_atoms, -1))
        heat_map_of_atoms[i,:,:] = np.sum(bin_weighted_atoms_to_use, axis=0).reshape((nb_rows,nb_columns))
        if show_dilated_atoms:
            bin_weighted_atoms = binarized_weighted_atoms(dilated_atoms, h_images[i], 
                                            nb_atoms_to_use_for_threshold_computation=nb_atoms_to_use_for_threshold_computation)
            bin_weighted_atoms_to_use = bin_weighted_atoms[idx_sort[:   number_of_atoms]].reshape((number_of_atoms, -1))
            heat_map_of_dilated_atoms[i,:,:] = np.sum(bin_weighted_atoms_to_use, axis=0).reshape((28,28))
    max_nb = max(np.max(heat_map_of_atoms), np.max(heat_map_of_dilated_atoms))
    plt.figure(figsize=(nb_samples_to_plot*3,n_rows_in_figure*3))
    fig, axes = plt.subplots(nrows=n_rows_in_figure, ncols=nb_samples_to_plot, figsize=(nb_samples_to_plot*2, n_rows_in_figure*2))
    i = 0
    for ax in axes.flat:
        ax.set_axis_off()
        if i<nb_samples_to_plot:
            im = ax.imshow(heat_map_of_atoms[i].reshape((nb_rows,nb_columns)), cmap='hot', vmin=0, vmax=max_nb)    
        if i>=nb_samples_to_plot:
            im = ax.imshow(heat_map_of_dilated_atoms[i-nb_samples_to_plot].reshape((nb_rows,nb_columns)), cmap='hot', vmin=0, vmax=max_nb)    
        i += 1
    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    plt.show()

def mean_number_overlapping_binarzed_weighted_atoms_at_each_pixel(atoms, h_test_image, nb_atoms_to_use_for_threshold_computation=25):
    bin_weighted_atoms = binarized_weighted_atoms(atoms, h_test_image, 
                                            nb_atoms_to_use_for_threshold_computation=nb_atoms_to_use_for_threshold_computation)[:,:,:,0]
    return np.mean(np.sum(bin_weighted_atoms, 0))

def mean_on_all_images_of_the_number_of_overlapping_binarzed_weighted_atoms_at_each_pixel(atoms, h_test, nb_atoms_to_use_for_threshold_computation=25):
    nb_samples, _ = h_test.shape
    s = 0
    for i in range(nb_samples):
        s += mean_number_overlapping_binarzed_weighted_atoms_at_each_pixel(atoms, h_test[i], nb_atoms_to_use_for_threshold_computation=nb_atoms_to_use_for_threshold_computation)
    return s/nb_samples

def Hoyer_sparsity_of_atoms(atoms):
    """
    Returns the mean of the sparsity measure of the set of atoms as defined in Hoyer 2004.
    The measure is equal to 0 iif all entries of a vector are equals, and 1 iif all but one are equal to zero.
    atoms: shape (nb_atoms, nb_rows, nb_columns, nb_channel)
    """
    _, nb_rows, nb_columns, nb_channel = atoms.shape 
    nb_pixels = nb_rows*nb_columns*nb_channel
    sqrt = math.sqrt(nb_pixels)
    vectorized_atoms = atoms.reshape((-1, nb_pixels))
    sigma = (sqrt - (np.linalg.norm(vectorized_atoms, ord=1, axis=1)/(np.linalg.norm(vectorized_atoms, ord=2, axis=1)+0.0000001)))/(sqrt - 1)
    return np.mean(sigma)