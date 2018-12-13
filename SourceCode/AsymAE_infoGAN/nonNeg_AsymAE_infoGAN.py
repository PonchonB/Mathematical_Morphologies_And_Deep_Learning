from __future__ import print_function
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Activation, Conv2D
from keras import losses, regularizers
import keras.utils 
import h5py
import math
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn import svm
import datetime
import matplotlib.pyplot as plt
import metrics
import custom_regularizers
from AsymAE_infoGAN import AsymAEinfoGAN

class NonNegAsymAEinfoGAN_Asymmetric_decay(AsymAEinfoGAN):
    """
    AsymAE with Non Negativity of the weights enforced by adding an asymetric weight decay to the total loss.
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
        decay_positive_weights: positive float - the weight decay parameter for the positive weights.
        decay_negative_weights: positive float - the weight decay parameter for the negative weights.
        decay_weight: positive float - the weight of the whole non negativity cost.
    """
    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True, 
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
        self.nb_rows=nb_rows
        self.nb_columns=nb_columns
        if one_channel_output:
            self.nb_output_channels=1
        else:
            self.nb_output_channels=nb_input_channels
        self.decay_positive_weights = decay_positive_weights        
        self.decay_negative_weights = decay_negative_weights
        self.decay_weight = decay_weight
        input_img = Input(shape=(self.nb_rows, self.nb_columns, self.nb_input_channels))  # adapt this if using `channels_first` image data format
        x = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(input_img)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(128, (4, 4), strides=(2,2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        encoded = Dense(self.latent_dim, activation='sigmoid')(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(self.nb_rows*self.nb_columns*self.nb_output_channels, kernel_regularizer=custom_regularizers.asymmetric_weight_decay(alpha=self.decay_positive_weights, 
                                                                                       beta=self.decay_negative_weights, 
                                                                                       lam=self.decay_weight))(encoded_img)
        x = LeakyReLU(alpha=0.1)(x)
        decoded = Reshape((self.nb_rows,self.nb_columns,self.nb_output_channels))(x)
        self.decoder = Model(encoded_img, decoded, name='decoder')
        encoded = self.encoder(input_img)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['mse'])
        
    @classmethod
    def load(cls, model_name, custom_objects={}, path_to_model_directory="../Results/AsymAE_infoGAN/"):
        """
        Load a autoencoder previously saved with the save method, or a model saved as a h5 file.
        The file is looked for in the directory path_to_model_directory/NonNegativity/Asym_Decay/Models/.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
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
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]     
        loaded_AE.decay_positive_weights = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['alpha']
        loaded_AE.decay_negative_weights = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['beta']
        loaded_AE.decay_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['lam']
        return loaded_AE

    def save(self, path_to_model_directory="../Results/AsymAE_infoGAN/", model_name=None):
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "NonNegativity/Asym_Decay/Models/"
        strDate = d.strftime("%y_%m_%d")
        model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_Asym_Decay_" + str(self.decay_positive_weights) + "_" + str(self.decay_negative_weights) + "_" + str(self.decay_weight) + ".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'decay_positive_weights':self.decay_positive_weights, 'decay_negative_weights':self.decay_negative_weights, 'decay_weight':self.decay_weight}


class NonNegAsymAEinfoGAN_NonNegConstraint(AsymAEinfoGAN):
    """
    AsymAE with Non Negativity of the weights enforced by adding an asymetric weight decay to the total loss.
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
        sparsity_weight: positive float, the weight of the sparsity cost function in the total loss of the network.
        sparsity_objective: float between 0 and 1, the value of the Bernouilli distribution considered as objective in the KL divergence.
    """
    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True):
        """
        Create a shallow AE with the Keras Non Negativity Constraint.
        """
        self.latent_dim = latent_dim
        self.nb_input_channels=nb_input_channels
        self.nb_rows=nb_rows
        self.nb_columns=nb_columns
        if one_channel_output:
            self.nb_output_channels=1
        else:
            self.nb_output_channels=nb_input_channels
        input_img = Input(shape=(self.nb_rows, self.nb_columns, self.nb_input_channels))  # adapt this if using `channels_first` image data format
        x = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(input_img)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(128, (4, 4), strides=(2,2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        encoded = Dense(self.latent_dim, activation='sigmoid')(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(self.nb_rows*self.nb_columns*self.nb_output_channels, kernel_constraint=constraints.non_neg())(encoded_img)
        x = LeakyReLU(alpha=0.1)(x)
        decoded = Reshape((self.nb_rows,self.nb_columns,self.nb_output_channels))(x)
        self.decoder = Model(encoded_img, decoded, name='decoder')
        encoded = self.encoder(input_img)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['mse'])
        
    @classmethod
    def load(cls, model_name, custom_objects={}, path_to_model_directory="../Results/AsymAE_infoGAN/"):
        """
        Load a autoencoder previously saved with the save method, or a model saved as a h5 file.
        The file is looked for in the directory path_to_model_directory/NonNegativity/NonNegConstraint/Models/.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "NonNegativity/NonNegConstraint/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=custom_objects)
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]      
        return loaded_AE

    def save(self, path_to_model_directory="../Results/AsymAE_infoGAN/", model_name=None):
        """
        Save the model as a h5 file under the following path: 
                path_to_model_directory/NonNegativity/NonNegativityConstraint/Models/yy_mm_dd_dim'latent_dim'_NonNegConstraint.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "NonNegativity/NonNegConstraint/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_NonNegConstraint.h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_NonNegConstraint" + model_name + ".h5"        
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {}
