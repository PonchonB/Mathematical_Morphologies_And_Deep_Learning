from __future__ import print_function
from shallowAE import ShallowAE
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Activation
from keras import losses, regularizers, metrics, constraints
import keras.utils 
import h5py
import custom_regularizers
import datetime

class NonNegShallowAE_Asymmetric_decay(ShallowAE):
    
    def __init__(self, latent_dim=100, nb_input_channels=1, one_channel_output=True, 
                decay_positive_weights=0, decay_negative_weights=1, decay_weight=1):
        """
        Create a shallow AE with a Non Negativity Constraint on the weights enforced with asymetric weight decay.
        Arguments:
            decay_positive_weights: positive float - the weight decay parameter for the positive weights.
            decay_negative_weights: positive float - the weight decay parameter for the negative weights.
            decay_weight: positive float - the weight of the whole non negativity cost.
        """
        self.latent_dim = latent_dim
        self.nb_input_channels=nb_input_channels
        if one_channel_output:
            self.nb_output_channels=1
        else:
            self.nb_output_channels=nb_input_channels
        self.decay_positive_weights = decay_positive_weights        
        self.decay_negative_weights = decay_negative_weights
        self.decay_weight = decay_weight
        input_img = Input(shape=(28, 28, nb_input_channels))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid',
                        kernel_regularizer=custom_regularizers.asymmetric_weight_decay(alpha=self.decay_positive_weights, 
                                                                                       beta=self.decay_negative_weights, 
                                                                                       lam=self.decay_weight))(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(28*28*self.nb_output_channels)(encoded_img)
        x = LeakyReLU(alpha=0.1)(x)
        decoded = Reshape((28,28,self.nb_output_channels))(x)
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
        path_to_directory = path_to_model_directory + "NonNegativity/Asym_Decay/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'asymmetric_weight_decay':custom_regularizers.asymmetric_weight_decay}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]   
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]     
        loaded_AE.decay_positive_weights = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['alpha']
        loaded_AE.decay_negative_weights = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['beta']
        loaded_AE.decay_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['lam']
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/"):
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "NonNegativity/Asym_Decay/Models/"
        strDate = d.strftime("%y_%m_%d")
        model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_Asym_Decay_" + str(self.decay_positive_weights) + "_" + str(self.decay_negative_weights) + "_" + str(self.decay_weight) + ".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'decay_positive_weights':self.decay_positive_weights, 'decay_negative_weights':self.decay_negative_weights, 'decay_weight':self.decay_weight}


class NonNegShallowAE_NonNegConstraint(ShallowAE):
    
    def __init__(self, latent_dim=100, nb_input_channels=1, one_channel_output=True):
        """
        Create a shallow AE with the Keras Non Negativity Constraint.
        """
        self.latent_dim = latent_dim
        self.nb_input_channels=nb_input_channels
        if one_channel_output:
            self.nb_output_channels=1
        else:
            self.nb_output_channels=nb_input_channels
        input_img = Input(shape=(28, 28, nb_input_channels))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', kernel_constraint=constraints.non_neg())(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(28*28*self.nb_output_channels)(encoded_img)
        x = LeakyReLU(alpha=0.1)(x)
        decoded = Reshape((28,28,self.nb_output_channels))(x)
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
        path_to_directory = path_to_model_directory + "NonNegativity/NonNegConstraint/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=custom_objects)
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]      
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/"):
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "NonNegativity/NonNegConstraint/Models/"
        strDate = d.strftime("%y_%m_%d")
        model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_NonNegConstraint.h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {}
