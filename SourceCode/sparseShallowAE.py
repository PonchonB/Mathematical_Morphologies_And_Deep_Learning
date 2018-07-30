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
    """
    Shallow encoders with sparsity enforced by minimizing the KL divergence of the mean over the batches 
        of each code coefficient to a bernouilli density with a given parameter.
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
    
    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True, 
                                       sparsity_weight=1, sparsity_objective=0.1):
        """
        Create a sparse shallow AE with the custom kl divergence regularizer. Multiple losses are computed for each code coefficient (KL_div + rec_err)
        """
        self.latent_dim = latent_dim
        self.nb_input_channels=nb_input_channels
        self.nb_rows=nb_rows
        self.nb_columns=nb_columns
        if one_channel_output:
            self.nb_output_channels=1
        else:
            self.nb_output_channels=nb_input_channels
        self.sparsity_weight = sparsity_weight
        self.sparsity_objective = sparsity_objective
        input_img = Input(shape=(self.nb_rows, self.nb_columns, nb_input_channels))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', 
                        activity_regularizer=custom_regularizers.KL_divergence(beta=self.sparsity_weight, 
                                                                                rho=self.sparsity_objective))(x)
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
        The file is looked for in the directory path_to_model_directory/Sparse/KL_div/Models/.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Sparse/KL_div/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'KL_divergence':custom_regularizers.KL_divergence}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]  
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]      
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['rho']
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/", model_name=None):
        """
        Save the model as a h5 file under the following path: 
                path_to_model_directory/Sparse/KL_div/Models/yy_mm_dd_dim'latent_dim'_KLdiv_'sparsity_weight'_'sparsity_objective'.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse/KL_div/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdiv_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + ".h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdiv_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_" + model_name + ".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective}


class SparseShallowAE_L1(ShallowAE):
    """
    Shallow encoders with sparsity enforced by minimizing the L1 norm of the code.
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
    """
    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True,
                        sparsity_weight=1):
        """
        Create a sparse shallow AE with the custom L1-norm regularizer.
        """
        self.latent_dim = latent_dim
        self.nb_input_channels=nb_input_channels
        if one_channel_output:
            self.nb_output_channels=1
        else:
            self.nb_output_channels=nb_input_channels
        self.nb_rows=nb_rows
        self.nb_columns=nb_columns
        self.sparsity_weight = sparsity_weight
        input_img = Input(shape=(self.nb_rows, self.nb_columns, nb_input_channels))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', 
                        activity_regularizer=custom_regularizers.L1(beta=self.sparsity_weight))(x)
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
        The file is looked for in the directory path_to_model_directory/Sparse/L1/Models.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Sparse/L1/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'L1':custom_regularizers.L1}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]    
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['l1']
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/", model_name=None):
        """
        Save the model as a h5 file under the following path: 
                path_to_model_directory/Sparse/L1/Models/yy_mm_dd_dim'latent_dim'_L1_'sparsity_weight'.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse/L1/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_L1_" + str(self.sparsity_weight) + ".h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_L1_" + str(self.sparsity_weight) + "_" + model_name + ".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight}

class SparseShallowAE_KL_sum(ShallowAE):
    """
    Shallow encoders with sparsity enforced by minimizing the KL divergence regularizer, 
        the actual one, with the KL divergence being sum over all code coefficients.
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

    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True, 
                    sparsity_weight=1, sparsity_objective=0.1):
        """
        Create a sparse shallow AE with the custom kl divergence regularizer. The actual KL divergence.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
            sparsity_objective: float between 0 and 1 - the sparsity parameter
        """
        self.latent_dim = latent_dim
        self.nb_rows=nb_rows
        self.nb_columns=nb_columns
        self.nb_input_channels=nb_input_channels
        if one_channel_output:
            self.nb_output_channels=1
        else:
            self.nb_output_channels=nb_input_channels
        self.sparsity_weight = sparsity_weight
        self.sparsity_objective = sparsity_objective
        input_img = Input(shape=(self.nb_rows, self.nb_columns, nb_input_channels))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', 
                        activity_regularizer=custom_regularizers.KL_divergence_sum(beta=self.sparsity_weight,
                                                                                    rho=self.sparsity_objective))(x)
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
        path_to_directory = path_to_model_directory + "Sparse/KL_div_sum/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'KL_divergence_sum':custom_regularizers.KL_divergence_sum}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]            
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['rho']
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/", model_name=None):
        """
        Save the model as a h5 file under the following path:
             path_to_model_directory/Sparse/KL_div_sum/Models/yy_mm_dd_dim'latent_dim'_KLdivSum_'sparsity_weight'_'sparsity_objective'.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse/KL_div_sum/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdivSum_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + ".h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdivSum_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_" + model_name + ".h5"        
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective}

