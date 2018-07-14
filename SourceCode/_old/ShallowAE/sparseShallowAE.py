from __future__ import print_function
from shallowAE import ShallowAE
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Activation
from keras import losses, regularizers, metrics
import keras.utils 
import h5py
import custom_regularizers
import datetime

class SparseShallowAE_KL(ShallowAE):
    
    def __init__(self, latent_dim=100, beta=1, rho=0.1):
        """
        Create a sparse shallow AE with the custom kl divergence regularizer.
        Arguments:
            beta: positive float - the weight of the sparsity cost.
            rho: float between 0 and 1 - the sparsity parameter
        """
        self.latent_dim = latent_dim
        self.beta = beta
        self.rho = rho
        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', activity_regularizer=custom_regularizers.KL_divergence(beta=self.beta, rho=self.rho))(x)
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
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Sparse/KL_div/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'KL_divergence':custom_regularizers.KL_divergence}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.beta = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['beta']
        loaded_AE.rho = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['rho']
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/"):
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse/KL_div/Models/"
        strDate = d.strftime("%y_%m_%d")
        model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdiv.h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'beta':self.beta, 'rho':self.rho}


class SparseShallowAE_L1(ShallowAE):
    
    def __init__(self, latent_dim=100, beta=1):
        """
        Create a sparse shallow AE with the custom kl divergence regularizer.
        Arguments:
            beta: positive float - the weight of the sparsity cost.
        """
        self.latent_dim = latent_dim
        self.beta = beta
        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', activity_regularizer=regularizers.l1(self.beta))(x)
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
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Sparse/L1/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=custom_objects)
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.beta = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['l1']
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/"):
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse/L1/Models/"
        strDate = d.strftime("%y_%m_%d")
        model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_L1.h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'beta':self.beta}
