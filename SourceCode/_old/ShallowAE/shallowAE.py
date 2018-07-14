from __future__ import print_function
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Activation
from keras import losses, regularizers, metrics
import keras.utils 
import h5py
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn import svm
import datetime
import matplotlib.pyplot as plt

class ShallowAE:

    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid')(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(28*28)(encoded_img)
        x = LeakyReLU(alpha=0.1)(x)
        decoded = Reshape((28,28,1))(x)
        self.decoder = Model(encoded_img, decoded, name='decoder')
        encoded = self.encoder(input_img)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
        
    @classmethod
    def load(cls, model_name, custom_objects={}, path_to_model_directory="../ShallowAE/"):
        """
        Load a autoencoder previously saved as a h5 file.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Simple/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=custom_objects)
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]
        return loaded_AE
        
    def train(self, X_train, nb_epochs=100, X_val=None, verbose=1):
        if X_val is None:
            validation_data = None
        else:
            validation_data = (X_val, X_val)
        self.autoencoder.fit(X_train, X_train,
                epochs=nb_epochs,
                batch_size=128,
                shuffle=True,
                validation_data=validation_data)
        
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_autoencoder(self):
        return self.autoencoder
    
    def encode(self, X_test):
        return self.encoder.predict(X_test)
    
    def decode(self, H_test):
        return self.decoder.predict(H_test)
    
    def reconstruction(self, X_test):
        return self.autoencoder.predict(X_test)
    
    def save(self, path_to_model_directory="../ShallowAE/"):
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Simple/Models/"
        strDate = d.strftime("%y_%m_%d")
        model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + ".h5"
        self.autoencoder.save(model_path)
        
    def reconstruction_error(self, X_test):
        return self.autoencoder.evaluate(X_test, X_test, verbose=0)
        
    def plot_reconstructions(self, X_test):
        """
        Plots the original images, as well as their reconstructions by the given autoencoder.

        Arguments:
            AE: an autoencoder.
            X: a numpy array of shape (10, 28, 28, 1)
        """
        X_rec = self.reconstruction(X_test)
        plt.figure(figsize=(20, 4))
        for i in range(10):
            # display original
            ax = plt.subplot(2, 10, i + 1)
            plt.imshow(X_test[i, :,:,0])
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
        
    def atom_images(self):
        """
        Return the atom images of a shallow encoder.
        """
        W= self.encoder.get_weights()[0]
        nbAtoms = W.shape[1]
        W = keras.utils.normalize(W, axis=0, order=2)
        atoms = W.reshape((28,28,nbAtoms))
        return atoms
    
    def plot_atoms(self, nb_to_plot=-1):
        """
        Plot the weights of a shallow encoder.

        Arguments: 
           encode: shallow encoder.
           nb_to_plot: number of basis images to plot, -1 is all, otherwise plot the nb_to_plot first ones.
        """
        atoms = self.atom_images()
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
        
    def best_SVM_classification_score(self, X, y, min_log_C=-2, max_log_C=3, nb_values_C=7, min_log_gamma=-3, max_log_gamma=2, nb_values_gamma=7):
        H = self.encode(X)
        H_train, H_test, Y_train, Y_test = train_test_split(H, y, test_size=0.1)
        C_range = np.logspace(min_log_C, max_log_C, nb_values_C)
        gamma_range = np.logspace(min_log_gamma, max_log_gamma, nb_values_gamma)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        grid = GridSearchCV(svm.SVC(cache_size=400), param_grid=param_grid, cv=cv, verbose=2)
        grid.fit(H_train, Y_train)
        return grid.score(H_test, Y_test)






