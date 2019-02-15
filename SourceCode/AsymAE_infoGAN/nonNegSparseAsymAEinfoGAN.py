from __future__ import print_function
from AsymAE_infoGAN.AsymAE_infoGAN import AsymAEinfoGAN
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Activation, Conv2D
from keras import losses, regularizers, constraints
import keras.utils 
import h5py
import custom_regularizers
import datetime

class Sparse_NonNeg_AsymAEinfoGAN_Hoyer_NonNegConstraint(AsymAEinfoGAN):
    """
    Auto-Encoder with sparse and Non-Negativity constraintes (Hoyer measure for sparsity, Keras NonNeg constraint (projection) for Non Negativity)
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
        sparsity_weight: positive float - the weight of the sparsity cost.
        sparsity_objective: float between 0 and 1 - the objective value for Hoyer metric, the greater the sparser.
    """
    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True, 
                    sparsity_weight=1, sparsity_objective=0.6, leakyReLU=True):
        """
        Create a sparse AsymAE (infoGAN encoder) AE with the custom Hoyer regularizer, enforcing weights non negativity with Keras NonNeg constraint.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
            sparsity_objective: float between 0 and 1 - the sparsity ojective.
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
        encoded = Dense(self.latent_dim, activation='sigmoid', 
                        activity_regularizer=custom_regularizers.sparsity_Hoyer(beta=self.sparsity_weight, 
                                                                                sp_o=self.sparsity_objective))(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(self.nb_rows*self.nb_columns*self.nb_output_channels, 
                    kernel_constraint=constraints.non_neg())(encoded_img)
        if leakyReLU:
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
        The file is looked for in the directory path_to_model_directory/Sparse_NonNeg/Hoyer_NonNegConstraint/Models/.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/Hoyer_NonNegConstraint/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'sparsity_Hoyer':custom_regularizers.sparsity_Hoyer}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]        
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['sp_o']
        loaded_AE.leakyReLU = (len(loaded_AE.decoder.layers) > 3)
        return loaded_AE

    def save(self, path_to_model_directory="../Results/AsymAE_infoGAN/", model_name=None):
        """
        Save the model as a h5 file under the following path: 
                    path_to_model_directory/Sparse_NonNeg/Hoyer_NonNegConstraint/Models/yy_mm_dd_dim'latent_dim'_KLdiv_'sparsity_weight'_'sparsity_objective'_NonNegConstraint.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/Hoyer_NonNegConstraint/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_Hoyer_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_NonNegConstraint.h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_Hoyer_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_NonNegConstraint_" + model_name +".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective}


class Sparse_NonNeg_AsymAEinfoGAN_KL_AsymDecay(AsymAEinfoGAN):
    """
    Auto-Encoder with sparse and Non-Negativity constraintes (KL divergence for sparsity, Asymetric Decay for Non Negativity)
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
        sparsity_weight: positive float - the weight of the sparsity cost.
        sparsity_objective: float between 0 and 1 - the sparsity parameter.
        decay_positive_weights: positive float - the weight decay parameter for the positive weights.
        decay_negative_weights: positive float - the weight decay parameter for the negative weights.
        decay_weight: positive float - the weight of the whole non negativity cost.
    """

    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True, 
                    sparsity_weight=1, sparsity_objective=0.1, 
                    decay_positive_weights=0, decay_negative_weights=1, decay_weight=1, leakyReLU=True): 
        """
        Create a sparse AsymAE (infoGAN encoder) AE with the custom kl divergence regularizer, enforcing weights non negativity with an asymmetric decay.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
            sparsity_objective: float between 0 and 1 - the sparsity parameter.
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
        self.sparsity_weight = sparsity_weight
        self.sparsity_objective = sparsity_objective
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
        encoded = Dense(self.latent_dim, activation='sigmoid', 
                        activity_regularizer=custom_regularizers.KL_divergence(beta=self.sparsity_weight, 
                                                                                rho=self.sparsity_objective))(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(self.nb_rows*self.nb_columns*self.nb_output_channels, 
                    kernel_regularizer=custom_regularizers.asymmetric_weight_decay(alpha=self.decay_positive_weights, 
                                                                                    beta=self.decay_negative_weights, 
                                                                                    lam=self.decay_weight))(encoded_img)
        if leakyReLU:
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
        The file is looked for in the directory path_to_model_directory/Sparse_NonNeg/KLdiv_AsymDecay/Models/.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """        
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdiv_AsymDecay/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'KL_divergence':custom_regularizers.KL_divergence, 'asymmetric_weight_decay':custom_regularizers.asymmetric_weight_decay}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]        
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['rho']
        loaded_AE.decay_positive_weights = loaded_AE.decoder.get_config()['layers'][1]['config']['kernel_regularizer']['config']['alpha']
        loaded_AE.decay_negative_weights = loaded_AE.decoder.get_config()['layers'][1]['config']['kernel_regularizer']['config']['beta']
        loaded_AE.decay_weight = loaded_AE.decoder.get_config()['layers'][1]['config']['kernel_regularizer']['config']['lam']
        loaded_AE.leakyReLU = (len(loaded_AE.decoder.layers) > 3)
        return loaded_AE

    def save(self, path_to_model_directory="../Results/AsymAE_infoGAN/", model_name=None):
        """
        Save the model as a h5 file under the following path: 
                    path_to_model_directory/Sparse_NonNeg/KLdiv_AsymDecay/Models/yy_mm_dd_dim'latent_dim'_KLdiv_'sparsity_weight'_'sparsity_objective'_AsymDecay_'decay_positive_weights'_'devay_negative_weights'_'decay_weight'.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdiv_AsymDecay/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdiv_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_AsymDecay_" + str(self.decay_positive_weights) + "_" + str(self.decay_negative_weights) + "_" + str(self.decay_weight) + ".h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdiv_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_AsymDecay_" + str(self.decay_positive_weights) + "_" + str(self.decay_negative_weights) + "_" + str(self.decay_weight) + "_" + model_name +".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective, 'decay_positive_weights':self.decay_positive_weights, 'decay_negative_weights':self.decay_negative_weights, 'decay_weight':self.decay_weight}


class Sparse_NonNeg_AsymAEinfoGAN_KL_NonNegConstraint(AsymAEinfoGAN):
    """
    Auto-Encoder with sparse and Non-Negativity constraintes (KL divergence for sparsity, Keras NonNeg constraint (projection) for Non Negativity)
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
        sparsity_weight: positive float - the weight of the sparsity cost.
        sparsity_objective: float between 0 and 1 - the sparsity parameter.
    """
    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True, 
                    sparsity_weight=1, sparsity_objective=0.1, leakyReLU=True): 
        """
        Create a sparse AsymAE (infoGAN encoder) AE with the custom kl divergence regularizer, enforcing weights non negativity with Keras NonNeg constraint.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
            sparsity_objective: float between 0 and 1 - the sparsity parameter.
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
        encoded = Dense(self.latent_dim, activation='sigmoid', 
                        activity_regularizer=custom_regularizers.KL_divergence(beta=self.sparsity_weight, 
                                                                                rho=self.sparsity_objective))(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(self.nb_rows*self.nb_columns*self.nb_output_channels, 
                    kernel_constraint=constraints.non_neg())(encoded_img)
        if leakyReLU:
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
        The file is looked for in the directory path_to_model_directory/Sparse_NonNeg/KLdiv_NonNegConstraint/Models/.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdiv_NonNegConstraint/Models/"
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
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['rho']
        loaded_AE.leakyReLU = (len(loaded_AE.decoder.layers) > 3)
        return loaded_AE

    def save(self, path_to_model_directory="../Results/AsymAE_infoGAN/", model_name=None):
        """
        Save the model as a h5 file under the following path: 
                    path_to_model_directory/Sparse_NonNeg/KLdiv_NonNegConstraint/Models/yy_mm_dd_dim'latent_dim'_KLdiv_'sparsity_weight'_'sparsity_objective'_NonNegConstraint.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdiv_NonNegConstraint/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdiv_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_NonNegConstraint.h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdiv_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_NonNegConstraint_" + model_name +".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective}


class Sparse_NonNeg_AsymAEinfoGAN_L1_AsymDecay(AsymAEinfoGAN):
    """
    Auto-Encoder with sparse and Non-Negativity constraints (L1-regularizer for sparsity, Asymmetric decay for Non Negativity)
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
        sparsity_weight: positive float - the weight of the sparsity cost.
        decay_positive_weights: positive float - the weight decay parameter for the positive weights.
        decay_negative_weights: positive float - the weight decay parameter for the negative weights.
        decay_weight: positive float - the weight of the whole non negativity cost.
    """

    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True, 
                    sparsity_weight=1, decay_positive_weights=0, decay_negative_weights=1, decay_weight=1, leakyReLU=True): 
        """
        Create a sparse AsymAE (infoGAN encoder) AE with the custom kl divergence regularizer, enforcing Non Negativity with asymmetric weight decay.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
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
        self.sparsity_weight = sparsity_weight
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
        encoded = Dense(self.latent_dim, activation='sigmoid', 
                        activity_regularizer=regularizers.l1(self.sparsity_weight))(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(self.nb_rows*self.nb_columns*self.nb_output_channels, 
                    kernel_regularizer=custom_regularizers.asymmetric_weight_decay(alpha=self.decay_positive_weights, 
                                                                                    beta=self.decay_negative_weights, 
                                                                                    lam=self.decay_weight))(encoded_img)
        if leakyReLU:
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
        The file is looked for in the directory path_to_model_directory/Sparse_NonNeg/L1_AsymDecay/Models/.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/L1_AsymDecay/Models/"
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
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['l1']
        loaded_AE.decay_positive_weights = loaded_AE.decoder.get_config()['layers'][1]['config']['kernel_regularizer']['config']['alpha']
        loaded_AE.decay_negative_weights = loaded_AE.decoder.get_config()['layers'][1]['config']['kernel_regularizer']['config']['beta']
        loaded_AE.decay_weight = loaded_AE.decoder.get_config()['layers'][1]['config']['kernel_regularizer']['config']['lam']
        loaded_AE.leakyReLU = (len(loaded_AE.decoder.layers) > 3)
        return loaded_AE

    def save(self, path_to_model_directory="../Results/AsymAE_infoGAN/", model_name=None):
        """
        Save the model as a h5 file under the following path: 
                    path_to_model_directory/Sparse_NonNeg/L1_AsymDecay/Models/yy_mm_dd_dim'latent_dim'_L1_'sparsity_weight'_AsymDecay_'decay_positive_weights'_'devay_negative_weights'_'decay_weight'.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/L1_AsymDecay/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_L1_" + str(self.sparsity_weight) + "_Asym_Decay_" + str(self.decay_positive_weights) + "_" + str(self.decay_negative_weights) + "_" + str(self.decay_weight) + ".h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_L1_" + str(self.sparsity_weight) + "_Asym_Decay_" + str(self.decay_positive_weights) + "_" + str(self.decay_negative_weights) + "_" + str(self.decay_weight) + "_" + model_name + ".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'decay_positive_weights':self.decay_positive_weights, 'decay_negative_weights':self.decay_negative_weights, 'decay_weight':self.decay_weight}


class Sparse_NonNeg_AsymAEinfoGAN_L1_NonNegConstraint(AsymAEinfoGAN):
    """
    Auto-Encoder with sparse and Non-Negativity constraints (L1-regularizer for sparsity, Keras NonNeg constraint for Non Negativity)
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
        sparsity_weight: positive float - the weight of the sparsity cost.
    """
    
    def __init__(self, nb_rows=28, nb_columns=28, latent_dim=100, nb_input_channels=1, one_channel_output=True, sparsity_weight=1, leakyReLU=True): 
        """
        Create a sparse AsymAE (infoGAN encoder) AE with the custom kl divergence regularizer.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
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
        encoded = Dense(self.latent_dim, activation='sigmoid', 
                        activity_regularizer=regularizers.l1(self.sparsity_weight))(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(self.nb_rows*self.nb_columns*self.nb_output_channels, 
                    kernel_constraint=constraints.non_neg())(encoded_img)
        if leakyReLU:
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
        The file is looked for in the directory path_to_model_directory/"Sparse_NonNeg/L1_NonNegConstraint/Models/.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/L1_NonNegConstraint/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=custom_objects)
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['l1']
        loaded_AE.leakyReLU = (len(loaded_AE.decoder.layers) > 3)
        return loaded_AE

    def save(self, path_to_model_directory="../Results/AsymAE_infoGAN/", model_name=None):
        """
        Save the model as a h5 file under the following path: 
                    path_to_model_directory/Sparse_NonNeg/L1_NonNegConstraint/Models/yy_mm_dd_dim'latent_dim'_L1_'sparsity_weight'_NonNeg_Constraint.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/L1_NonNegConstraint/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_L1_" + str(self.sparsity_weight) + "_NonNegConstraint.h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_L1_" + str(self.sparsity_weight) + "_NonNegConstraint_" + model_name + ".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight}


class Sparse_NonNeg_AsymAEinfoGAN_KLsum_AsymDecay(AsymAEinfoGAN):
    """
    Auto-Encoder with sparse and Non-Negativity constraints (KL_sum for sparsity, Asymmetric decay for Non Negativity)
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
        sparsity_weight: positive float - the weight of the sparsity cost.
        sparsity_objective: float between 0 and 1 - the sparsity parameter.
        decay_positive_weights: positive float - the weight decay parameter for the positive weights.
        decay_negative_weights: positive float - the weight decay parameter for the negative weights.
        decay_weight: positive float - the weight of the whole non negativity cost.
    """

    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True, 
                    sparsity_weight=0.1, sparsity_objective=0.1, decay_positive_weights=0, decay_negative_weights=1, decay_weight=1, leakyReLU=True): 
        """
        Create a sparse AsymAE (infoGAN encoder) AE with the custom kl divergence regularizer, enforcing weights non negativity with an asymmetric decay.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
            sparsity_objective: float between 0 and 1 - the sparsity parameter.
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
        self.sparsity_weight = sparsity_weight
        self.sparsity_objective = sparsity_objective
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
        encoded = Dense(self.latent_dim, activation='sigmoid', 
                        activity_regularizer=custom_regularizers.KL_divergence_sum(beta=self.sparsity_weight, 
                                                                                        rho=self.sparsity_objective))(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(self.nb_rows*self.nb_columns*self.nb_output_channels, 
                    kernel_regularizer=custom_regularizers.asymmetric_weight_decay(alpha=self.decay_positive_weights, 
                                                                                    beta=self.decay_negative_weights, 
                                                                                    lam=self.decay_weight))(encoded_img)
        if leakyReLU:
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
        The file is looked for in the directory path_to_model_directory/Sparse_NonNeg/KLdivSum_AsymDecay/Models/.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdivSum_AsymDecay/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'KL_divergence_sum':custom_regularizers.KL_divergence_sum, 'asymmetric_weight_decay':custom_regularizers.asymmetric_weight_decay}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['rho']
        loaded_AE.decay_positive_weights = loaded_AE.decoder.get_config()['layers'][1]['config']['kernel_regularizer']['config']['alpha']
        loaded_AE.decay_negative_weights = loaded_AE.decoder.get_config()['layers'][1]['config']['kernel_regularizer']['config']['beta']
        loaded_AE.decay_weight = loaded_AE.decoder.get_config()['layers'][1]['config']['kernel_regularizer']['config']['lam']
        loaded_AE.leakyReLU = (len(loaded_AE.decoder.layers) > 3)
        return loaded_AE

    def save(self, path_to_model_directory="../Results/AsymAE_infoGAN/", model_name=None):
        """
        Save the model as a h5 file under the following path: 
                    path_to_model_directory/Sparse_NonNeg/KLdivSum_AsymDecay/Models/yy_mm_dd_dim'latent_dim'_KLdivSum_'sparsity_weight'_'sparsity_objective'_AsymDecay_'decay_positive_weights'_'devay_negative_weights'_'decay_weight'.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdivSum_AsymDecay/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdivSum_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_AsymDecay_" + str(self.decay_positive_weights) + "_" + str(self.decay_negative_weights) + "_" + str(self.decay_weight) + ".h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdivSum_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_AsymDecay_" + str(self.decay_positive_weights) + "_" + str(self.decay_negative_weights) + "_" + str(self.decay_weight) + "_" + model_name +".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective, 'decay_positive_weights':self.decay_positive_weights, 'decay_negative_weights':self.decay_negative_weights, 'decay_weight':self.decay_weight}


class Sparse_NonNeg_AsymAEinfoGAN_KLsum_NonNegConstraint(AsymAEinfoGAN):
    """
    Auto-Encoder with sparse and Non-Negativity constraints (KL_sum for sparsity, Keras Non Constraint for Non Negativity)
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
        sparsity_weight: positive float - the weight of the sparsity cost.
        sparsity_objective: float between 0 and 1 - the sparsity parameter.
    """

    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True, 
                    sparsity_weight=0.1, sparsity_objective=0.1, leakyReLU=True): 
        """
        Create a sparse AsymAE (infoGAN encoder) AE with the custom kl divergence regularizer, enforcing weights non negativity with Keras NonNeg constraint.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
            sparsity_objective: float between 0 and 1 - the sparsity parameter.
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
        encoded = Dense(self.latent_dim, activation='sigmoid', 
                        activity_regularizer=custom_regularizers.KL_divergence_sum(beta=self.sparsity_weight, 
                                                                                        rho=self.sparsity_objective))(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(self.nb_rows*self.nb_columns*self.nb_output_channels, 
                    kernel_constraint=constraints.non_neg())(encoded_img)
        if leakyReLU:
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
        The file is looked for in the directory path_to_model_directory/Sparse_NonNeg/KLdivSum_NonNegConstraint/Models/.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdivSum_NonNegConstraint/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'KL_divergence_sum':custom_regularizers.KL_divergence_sum}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['rho']
        loaded_AE.leakyReLU = (len(loaded_AE.decoder.layers) > 3)
        return loaded_AE

    def save(self, path_to_model_directory="../Results/AsymAE_infoGAN/", model_name=None):
        """
        Save the model as a h5 file under the following path: 
                    path_to_model_directory/Sparse_NonNeg/KLdivSum_NonNegConstraint/Models/yy_mm_dd_dim'latent_dim'_KLdivSum_'sparsity_weight'_'sparsity_objective'_NonNeg_Constraint.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdivSum_NonNegConstraint/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdivSum_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_NonNegConstraint.h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdivSum_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_NonNegConstraint" + model_name + ".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective}

class Sparse_NonNeg_AsymAEinfoGAN_KLsum_NonNegConstraint_L1WeightsDecay(AsymAEinfoGAN):
    """
    Auto-Encoder with sparse and Non-Negativity constraints (KL_sum for sparsity, Keras Non Constraint for Non Negativity, L1 weight decay for sparsity of the atoms)
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
        sparsity_weight: positive float - the weight of the sparsity cost.
        sparsity_objective: float between 0 and 1 - the sparsity parameter.
    """

    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True, 
                    sparsity_weight=0.1, sparsity_objective=0.1, l1_weight_decay = 0.1, leakyReLU=True): 
        """
        Create a sparse AsymAE (infoGAN encoder) AE with the custom kl divergence regularizer, enforcing weights non negativity with Keras NonNeg constraint.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
            sparsity_objective: float between 0 and 1 - the sparsity parameter.
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
        self.l1_weight_decay = l1_weight_decay
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
        encoded = Dense(self.latent_dim, activation='sigmoid', 
                        activity_regularizer=custom_regularizers.KL_divergence_sum(beta=self.sparsity_weight, 
                                                                                        rho=self.sparsity_objective))(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))  
        x = Dense(self.nb_rows*self.nb_columns*self.nb_output_channels, 
                    kernel_constraint=constraints.non_neg(), kernel_regularizer=regularizers.l1(l=l1_weight_decay))(encoded_img)
        if leakyReLU:
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
        The file is looked for in the directory path_to_model_directory/Sparse_NonNeg/KLdivSum_NonNegConstraint/Models/.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdivSum_NonNegConstraint/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'KL_divergence_sum':custom_regularizers.KL_divergence_sum}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][-1]['config']['activity_regularizer']['config']['rho']
        loaded_AE.l1_weight_decay = AE.decoder.get_config()['layers'][1]['config']['kernel_regularizer']['config']['l1']
        loaded_AE.leakyReLU = (len(loaded_AE.decoder.layers) > 3)
        return loaded_AE

    def save(self, path_to_model_directory="../Results/AsymAE_infoGAN/", model_name=None):
        """
        Save the model as a h5 file under the following path: 
                    path_to_model_directory/Sparse_NonNeg/KLdivSum_NonNegConstraint/Models/yy_mm_dd_dim'latent_dim'_KLdivSum_'sparsity_weight'_'sparsity_objective'_NonNeg_Constraint.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdivSum_NonNegConstraint_L1WeightDecay/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdivSum_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_NonNegConstraint.h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdivSum_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_NonNegConstraint" + model_name + ".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective, 'l1_weight_decay':self}
