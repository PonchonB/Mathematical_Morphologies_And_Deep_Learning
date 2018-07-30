from __future__ import print_function
from keras.models import load_model
from keras.models import Model
import keras.utils 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import h5py
from matplotlib import offsetbox
from sklearn.svm import SVC
from sklearn.model_selection import KFold

import sys
sys.path.append('../fashion_mnist')
from utils import mnist_reader

def load_data(filePath, train=True, test=True, subsetTest=False, subsetSize=10):
    """
    Load the FashionMNIST data set. 
    filePath is the path to the parent directory of the fashion_mnist local repository.
    If train=True, returns the training set (x_train and y_train two first items in the list).
    If test=True, returns the full test set (x_test and y_test two last items in the list). 
    If test=False, returns a small subset of the test set, of size subsetSize and with a balance number of images from each class (one from each with  subsetSize = 10)
    """
    ret=[]    
    if (train==True):
        x_train, y_train = mnist_reader.load_mnist(filePath + 'fashion_mnist/data/fashion', kind='train')
        x_train = x_train.astype('float32') / 255.
        x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
        ret.append(x_train)
        ret.append(y_train)
    x_test, y_test = mnist_reader.load_mnist(filePath + 'fashion_mnist/data/fashion', kind='t10k')
    x_test = x_test.astype('float32') / 255.
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
    if (test==True):    
        ret.append(x_test)
        ret.append(y_test)
    if subsetTest:
        x_test_small = np.zeros((subsetSize, 28,28,1))
        y_test_small = np.zeros(subsetSize)
        c_len = [x_test[y_test==j].shape[0] for j in range(10)]
        for i in range(subsetSize):
            c=i%10
            x_test_small[i, :,:,:]=np.copy(x_test[y_test==c][np.random.randint(c_len[c])])
            y_test_small[i] = c
        ret.append(x_test_small)
        ret.append(y_test_small)
    return ret


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(20, 20))
    ax = plt.subplot(111)
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 2e-4:
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            props=dict(boxstyle='round', edgecolor='white')
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(x_test[i,:,:,0], cmap=plt.cm.gray, zoom=1), X[i], bboxprops=props)
            ax.add_artist(imagebox)
    if title is not None:
        plt.title(title)


#### The following function are deprecated, equivalents exist in the ShallowAE class
 
def load_AE(path, custom_objects={}):
    """
    Load a autoencoder previously saved as a h5 file.
    Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
    Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
    """
    autoencoder = load_model(path, custom_objects=custom_objects)
    encoder = autoencoder.layers[1]
    decoder = autoencoder.layers[2]
    return autoencoder, encoder, decoder

def atom_images(encoder):
    """
    Return the atom images of a shallow encoder.
    """
    W= encoder.get_weights()[0]
    nbAtoms = W.shape[1]
    W = keras.utils.normalize(W, axis=0, order=2)
    atoms = W.reshape((28,28,nbAtoms))
    return atoms

def plot_atoms(encoder, nb_to_plot=-1):
    """
    Plot the weights of a shallow encoder.
    
    Arguments: 
       encode: shallow encoder.
       nb_to_plot: number of basis images to plot, -1 is all, otherwise plot the nb_to_plot first ones.
    """
    atoms = atom_images(encoder)
    if (nb_to_plot<0):
        n_atoms = atoms.shape[2]
    else:
        n_atoms=nb_to_plot
    n_columns = min(10, n_atoms)
    n_rows = int(n_atoms/10) +1   
    plt.figure(figsize=(n_columns*2,n_rows*2))
    for i in range(n_atoms):
        ax = plt.subplot(n_rows, n_columns, i + 1)
        plt.imshow(atoms[:, :,i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def crossValSVM(X, y, nb_fold=10):
    nb_sample, nb_features = X.shape
    kf = KFold(n_splits=nb_fold, shuffle=True)  
    svm_scores = np.zeros(nb_fold)
    i = 0
    for train_index, test_index in kf.split(X):        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svm = SVC()
        svm.fit(X_train, y_train)
        svm_scores[i] = svm.score(X_test, y_test)
        i+=1
    return np.mean(svm_scores)

def plot_reconstructions(AE, X):
    """
    Plots the original images, as well as their reconstructions by the given autoencoder.

    Arguments:
        AE: an autoencoder.
        X: a numpy array of shape (10, 28, 28, 1)
    """
    X_rec = AE.predict(X)
    plt.figure(figsize=(20, 4))
    for i in range(10):
        # display original
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(X[i, :,:,0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
  
        # display reconstruction
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(X_rec[i,:,:,0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)    
    plt.show()
