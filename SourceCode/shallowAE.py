from __future__ import print_function
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Activation
from keras import losses, regularizers, metrics
import keras.utils 
import h5py
import math
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn import svm
import datetime
import matplotlib.pyplot as plt
from sklearn.svm import SVC

class ShallowAE:
    """
    General class for shallow encoders (one layer encoder and one layer decoder).
    Attributes: 
        latent_dim: int, size of the latent code
        nb_rows: int, expected number of rows of the input images
        nb_columns:int, expected number of columns of the input images
        nb_input_channels: int, expected number of channels of the input images
        nb_output_channels: int, number of channels in the output of the autoencoder. 
                            It can only take two values: either one (if one_channel_output=True), or nb_input_channels (else)
        encoder: Model from keras.model
        decoder: Model from keras.model
        autoencoder: Model from keras.model, composed of two layers, the two previous attributes
    """

    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True):
        """
        Create and initialize an autoencoder.
        """
        self.latent_dim = latent_dim
        self.nb_input_channels=nb_input_channels
        self.nb_rows=nb_rows
        self.nb_columns=nb_columns
        if one_channel_output:
            self.nb_output_channels=1
        else:
            self.nb_output_channels=nb_input_channels
        input_img = Input(shape=(self.nb_rows, self.nb_columns, nb_input_channels))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid')(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(self.nb_rows*self.nb_columns*self.nb_output_channels)(encoded_img)
        x = LeakyReLU(alpha=0.1)(x)
        decoded = Reshape((self.nb_rows,self.nb_columns,self.nb_output_channels))(x)
        self.decoder = Model(encoded_img, decoded, name='decoder')
        encoded = self.encoder(input_img)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['mse'])
        
    @classmethod
    def load(cls, model_name, custom_objects={}, path_to_model_directory="../ShallowAE/"):
        """
        Load a autoencoder previously saved with the save method, or a model saved as a h5 file.
        The file is looked for in the directory path_to_model_directory/Simple/Models/.
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
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]
        return loaded_AE
        
    def train(self, X_train, X_train_expected_output=None, nb_epochs=100, X_val=None, verbose=1):
        """
        Train the autoencoder.
        Arguments:
            X_train: numpy array (nb_samples, nb_rows, nb_columns, nb_inputs_channels) the training input.
            X_train_expected_output: numpy array (nb_samples, nb_rows, nb_columns, nb_output_channels) the training output.
                                    If None, X_train is used instead. Either all channels (if nb_input_channels=nb_output_channels) or the first channel of X_train. 
            nb_epochs: int
            X_val: None, (X_val, Y_val) or numpy array (nb_samples, nb_rows, nb_columns, nb_input_channels) validation data, if specified, not taken account for the training, but the loss is returned at each epoch if verbose >0.
                    If only one array is specified then it is used for bothe validation input and the validation output. 
                    All channels are used for the validation output if nb_input_channels=nb_output_channels, otherwise the first channel of the validation input is used as output.
            verbose: 0, 1 or 2. 0: no prints. 1: progress bar. 2: only one line per epoch.                 
        """
        if X_val is None:
            validation_data = None
            cb = None
        elif type(X_val) is tuple:
            validation_data=X_val
        else:
            if (self.nb_output_channels==self.nb_input_channels):
                Y_val = X_val
            else:
                Y_val = X_val[:,:,:,0].reshape((X_val.shape[0], self.nb_rows, self.nb_columns, 1))
            validation_data = (X_val, Y_val)
        if X_train_expected_output is None:
            if (self.nb_output_channels==self.nb_input_channels):
                Y_train = X_train
            else:
                Y_train = X_train[:,:,:,0].reshape((X_train.shape[0], self.nb_rows, self.nb_columns, 1)) 
        else:
            Y_train = X_train_expected_output
        self.autoencoder.fit(X_train, Y_train,
                epochs=nb_epochs,
                batch_size=128,
                shuffle=True,
                validation_data=validation_data, 
                verbose=verbose)
                        
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
    
    def save(self, path_to_model_directory="../ShallowAE/", model_name=None):
        """
        Save the model as a h5 file under the following path: path_to_model_directory/Simmple/Models/yy_mm_dd_dim'latent_dim'.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Simple/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + ".h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_" + model_name + ".h5"
        self.autoencoder.save(model_path)
        
    def loss_and_mse(self, X_test, X_rec_th=None):
        """
        Returns a tuple with both the total loss and the mse between the reconstruction and the exected output.
        X_test: numpy array (nb_samples, nb_rows, nb_columns, nb_input_channels). Input of the autoencoder.
        X_rec_th: None or numpy array (nb_samples, nb_rows, nb_columns, nb_output_channels).
                    Expected reconstruction by the autoencoder. 
                    If None, X_test is used in place: all channels if nb_input_channels=nb_output_channels, first channel else. 
        """
        if X_rec_th is None:
            if (self.nb_output_channels==self.nb_input_channels):
                X_rec = X_test
            else:
                X_rec = X_test[:,:,:,0].reshape((X_test.shape[0], self.nb_rows, self.nb_columns, 1)) 
        else:
            X_rec = X_rec_th
        return self.autoencoder.evaluate(X_test, X_rec, verbose=0, batch_size=128)


    def reconstruction_error(self, X_test, X_rec_th=None):
        """
        Returns the mse between the reconstruction and the exected output.
        X_test: numpy array (nb_samples, nb_rows, nb_columns, nb_input_channels). Input of the autoencoder.
        X_rec_th: None or numpy array (nb_samples, nb_rows, nb_columns, nb_output_channels).
                    Expected reconstruction by the autoencoder. 
                    If None, X_test is used in place: all channels if nb_input_channels=nb_output_channels, first channel else. 
        """
        loss_and_mse = self.loss_and_mse(X_test, X_rec_th=X_rec_th) 
        try:
            mse=loss_and_mse[1]
        except IndexError:
            mse=loss_and_mse
        return mse
        
    def total_loss(self, X_test, X_rec_th=None):
        """
        Returns the total loss (mse + additional costs (constraints and regularizers)) between the reconstruction and the exected output.
        X_test: numpy array (nb_samples, nb_rows, nb_columns, nb_input_channels). Input of the autoencoder.
        X_rec_th: None or numpy array (nb_samples, nb_rows, nb_columns, nb_output_channels).
                    Expected reconstruction by the autoencoder. 
                    If None, X_test is used in place: all channels if nb_input_channels=nb_output_channels, first channel else. 
        """
        loss_and_mse = self.loss_and_mse(X_test, X_rec_th=X_rec_th) 
        try:
            total_loss=loss_and_mse[0]
        except IndexError:
            total_loss=loss_and_mse
        return total_loss

    def plot_reconstructions(self, X_test, channel_to_plot=0):
        """
        Plots the original images, as well as their reconstructions by the autoencoder.

        Arguments:
            X_test: a numpy array of shape (nb_samples, nb_rows, nb_columns, nb_input_channels)
                    Only the 10 first samples will be plotted.
            channel_to_plot: int, the output_channel that will be plotted.
        """
        if (channel_to_plot < self.nb_output_channels):
            X_rec = self.reconstruction(X_test)[:,:,:,channel_to_plot]
        else:
            print('Too big channel number...plotting channel 0 instead...')
            channel_to_plot=0
            X_rec = self.reconstruction(X_test)[:10,:,:,0]
        plt.figure(figsize=(20, 4))
        for i in range(10):
            # display original
            ax = plt.subplot(2, 10, i + 1)
            plt.imshow(X_test[i, :,:, channel_to_plot])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, 10, i + 1 + 10)
            plt.imshow(X_rec[i,:,:])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)    
        plt.show()
        
    def atom_images_encoder(self, normalize=True):
        """
        Return the weigths of the encoder as latent_dim * nb_input_channels atom images.
        The result is of shape (nb_rows, nb_columns, nb_channels, nb_atoms)
        Arguments:
            normalize: bool. If True each image is normalized, giving the artificial input images that maximize each of the code coefficients (with unity energy). 
        """
        W= self.encoder.get_weights()[0]
        nbAtoms = W.shape[1]
        W = np.reshape(W, (self.nb_rows*self.nb_columns, self.nb_input_channels, nbAtoms))
        if normalize:
            W = keras.utils.normalize(W, axis=0, order=2)
        atoms = W.reshape((self.nb_rows,self.nb_columns, self.nb_input_channels, nbAtoms))
        return atoms

    def atom_images_decoder(self, add_bias=False, normalize=False):
        """
        Returns the weights of the decoder as latent_dim * nb_output_channels atom images.
        The result is of shape (nb_atoms, nb_rows, nb_columns, nb_channels)
        Arguments:
            Arguments:
            normalize: bool. If True each image is normalized, giving the artificial input images that maximize each of the code coefficients (with unity energy). 
            add_bias: bool. If True, adds the bias to each images. The results then correspond to the input of the activation layer when all code coefficients are equal to one.

        """
        W= self.decoder.get_weights()[0]
        if add_bias:
            W = W + self.decoder.get_weights()[1] 
        W = np.reshape(W, (self.latent_dim, self.nb_rows*self.nb_columns, self.nb_output_channels))
        if normalize:
            W = keras.utils.normalize(W, axis=1, order=2)
        atoms = W.reshape((self.latent_dim, self.nb_rows, self.nb_columns, self.nb_output_channels))
        return atoms
    
    def plot_atoms_encoder(self, channel_to_plot=0, nb_to_plot=-1, normalize=True):
        """
        Plot the weights of the encoder.
        Arguments:
           normalize: bool. If True each image is normalized, giving the artificial input images that maximize each of the code coefficients (with unity energy). 
           nb_to_plot: number of basis images to plot, -1 is all, otherwise plot the nb_to_plot first ones.
           channel: channel to plot (there are nb_input_channels*nb_atoms atoms)
        """
        if (channel_to_plot < self.nb_input_channels):
            atoms = self.atom_images_encoder(normalize=normalize)[:,:,channel_to_plot, :]
        else:
            print('Too big channel number...plotting channel 0 instead...')
            channel_to_plot=0
            atoms = self.atom_images_encoder()[:,:,0, :]
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

    def plot_atoms_decoder(self, channel_to_plot=0, nb_to_plot=-1, add_bias=False, normalize=False):
        """
        Plot the weights of the decoder.
        Arguments:
            nb_to_plot: number of basis images to plot, -1 is all, otherwise plot the nb_to_plot first ones.
            channel: channel to plot (there are nb_input_channels*nb_atoms atoms)
            add_bias: bool, whether to add the bias (784,) to the weights.
            normalize: bool. If True each image is normalized, giving the artificial input images that maximize each of the code coefficients (with unity energy). 
        """
        if (channel_to_plot < self.nb_output_channels):
            atoms = self.atom_images_decoder(add_bias=add_bias, normalize=normalize)[:,:,:,channel_to_plot]
        else:
            print('Too big channel number...plotting channel 0 instead...')
            channel_to_plot=0
            atoms = self.atom_images_decoder(add_bias=add_bias)[:,:,:,0]
        if (nb_to_plot<0):
            n_atoms =self.latent_dim
        else:
            n_atoms=nb_to_plot
        n_columns = min(10, n_atoms)
        n_rows = int(n_atoms/10) +1   
        plt.figure(figsize=(n_columns*2,n_rows*2))
        for i in range(n_atoms):
            ax = plt.subplot(n_rows, n_columns, i + 1)
            plt.imshow(atoms[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        
    def best_SVM_classification_score(self, X, y, min_log_C=-2, max_log_C=3, nb_values_C=7, min_log_gamma=-3, max_log_gamma=2, nb_values_gamma=7):
        """
        Performs cross-validation and parameter seletion (grid search on the specified parameter) 
            and returns the best classification score using a SVM classifier with a gaussian kernel (rbf).
        Returns also the best parameters on the grid search, as a dictionary.
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
        H = self.encode(X)
        H_train, H_test, Y_train, Y_test = train_test_split(H, y, test_size=0.1)
        C_range = np.logspace(min_log_C, max_log_C, nb_values_C)
        gamma_range = np.logspace(min_log_gamma, max_log_gamma, nb_values_gamma)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        grid = GridSearchCV(svm.SVC(cache_size=600), param_grid=param_grid, cv=cv, verbose=2)
        grid.fit(H_train, Y_train)
        return grid.score(H_test, Y_test), grid.best_params_

    def svm_classifiation_score(self, X, y, C=1.0, kernel='rbf', gamma='auto', degree=3):
        """
        Returns the classifiation score on the learnt encoding using a SVM with specified parameters.
        The data X and the labels y are splitted into training and tesing sets.
        """
        H =self.encode(X)
        H_train, H_test, Y_train, Y_test = train_test_split(H, y, test_size=0.1)
        svm = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, cache_size=600)
        svm.fit(H_train, Y_train)
        return svm.score(H_test, Y_test)

    def apply_operator_to_decoder_atoms(self, operator, apply_to_bias=False, **kwargs):
        """
        Returns a new instance of ShallowAE where the the operator (with the arguments specified in **kwargs) had been applied to the atom images of the decoder.
        Regularizers are not included in the new model.
        """
        W = self.atom_images_decoder(add_bias=False, normalize=False)
        b = self.decoder.get_weights()[1]
        W_op = np.copy(W)
        if apply_to_bias:
            b_op = b.reshape((self.nb_rows, self.nb_columns, self.nb_output_channels))
        else:
            b_op= np.copy(b)
        for i in range(self.nb_output_channels):
            if apply_to_bias:
                b_op[:,:,i] = operator(b_op[:,:,i], **kwargs)
            for j in range(self.latent_dim):
                W_op[j,:,:,i]=operator(W[j,:,:,i], **kwargs)
        AE = ShallowAE(latent_dim=self.latent_dim, nb_rows=self.nb_rows, nb_columns=self.nb_columns,  
                        nb_input_channels=self.nb_input_channels, one_channel_output=(self.nb_output_channels==1))
        W = self.encoder.get_weights()
        AE.encoder.set_weights([np.copy(W[0]), np.copy(W[1])])
        W_op = np.reshape(W_op, (self.latent_dim, self.nb_rows*self.nb_columns*self.nb_output_channels))
        if apply_to_bias:
            b_op=b_op.flatten()
        AE.decoder.set_weights([W_op, b_op])
        return AE
        
    def max_approximation_error(self, X, operator, apply_to_bias=False, **kwargs):
        """
        Computes the mse between the application of the operator to the images of X_rec (n_samples, n_rows, n_columns), 
        the application of the operator to the reconstructions of the images of X_rec by the autoencoder, and the decoding after 
        applying the operator to the atoms of the decoder.
        Arguments:
            X: Numpy array of shape (n_samples, n_rows, n_columns, nb_channels), the first channel must be the original image
            operator: A callable function, that takes an image as input.
            **kwargs: Other arguments of the operator.
        """ 
        nb_samples = X.shape[0]
        def apply_operator_to_all_images(X):
            result = np.zeros((nb_samples, self.nb_rows, self.nb_columns, self.nb_output_channels))
            for i in range(nb_samples):
                for j in range(self.nb_output_channels):
                    result[i,:,:,j]= operator(X[i, :,:,j], **kwargs)
            return result
        AE_op = self.apply_operator_to_decoder_atoms(operator, apply_to_bias=apply_to_bias, **kwargs)
        X_rec = self.reconstruction(X)
        X_op = apply_operator_to_all_images(X)
        X_rec_op = apply_operator_to_all_images(X_rec)
        error_To_Original = AE_op.reconstruction_error(X, X_rec_th=X_op)
        error_To_Reconstruction = AE_op.reconstruction_error(X, X_rec_th=X_rec_op)
        return error_To_Original, error_To_Reconstruction

    def sparsity_measure(self, X):
        """
        Returns the sparsity measure defined by Hoyer 2004 applied to the encoding of X by the encoder.
        This measure can be used to compare the sparsity of the various encodings.
        Arguments:
            X: numpy array (nb_samples, nb_rows, nb_columns, nb_input_channels)
        """
        H = self.encode(X)
        sqrt = math.sqrt(self.latent_dim)
        sigma = (sqrt - (np.linalg.norm(H, ord=1, axis=1)/np.linalg.norm(H, ord=2, axis=1)+0.0000001))/(sqrt - 1)
        return np.mean(sigma)

    def plot_histograms_of_the_encoding(self, X):
        """
        Plots the histogram of the encoding of each of the images in X (nb_samples, nb_rows, nb_columns, nb_channels)
        """
        H = self.encode(X)
        plt.figure(figsize=(30, 4))
        for i in range(10):
            ax = plt.subplot(1, 10, i + 1)
            ax.hist(H[i], bins=self.latent_dim)
        plt.show()





