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
import sys
sys.path.append('../../')
import metrics
from shallowAE import ShallowAE
from MaxPlusDecoder.max_plus_dense_layer import MaxPlusDense
import custom_constraints
import custom_regularizers

class Sparse_NonNeg_ShallowAE_MaxPlus_KLsum_Between0and1Constraint(ShallowAE):
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

    def __init__(self, latent_dim=100, nb_rows=28, nb_columns=28, nb_input_channels=1, one_channel_output=True,
                    sparsity_weight=0.1, sparsity_objective=0.1):
        """
        Create and initialize an autoencoder.
        """
        self.latent_dim = latent_dim
        self.nb_input_channels=nb_input_channels
        self.nb_rows=nb_rows
        self.nb_columns=nb_columns
        self.sparsity_weight = sparsity_weight
        self.sparsity_objective = sparsity_objective
        if one_channel_output:
            self.nb_output_channels=1
        else:
            self.nb_output_channels=nb_input_channels
        input_img = Input(shape=(self.nb_rows, self.nb_columns, nb_input_channels))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', 
                        activity_regularizer=custom_regularizers.KL_divergence_sum(beta=self.sparsity_weight, 
                                                                                    rho=self.sparsity_objective))(x)
        self.encoder = Model(input_img, encoded, name='encoder')
        encoded_img = Input(shape=(self.latent_dim,))
        x = MaxPlusDense(self.nb_rows*self.nb_columns*self.nb_output_channels, use_bias=False,
                            kernel_constraint=custom_constraints.Between_0_and_1())(encoded_img)
        decoded = Reshape((self.nb_rows,self.nb_columns,self.nb_output_channels))(x)
        self.decoder = Model(encoded_img, decoded, name='decoder')  
        encoded = self.encoder(input_img)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['mse'])
        
    @classmethod
    def load(cls, model_name, custom_objects={}, path_to_model_directory="../Results/ShallowAE_MaxPlus/"):
        """
        Load a autoencoder previously saved with the save method, or a model saved as a h5 file.
        The file is looked for in the directory path_to_model_directory/Simple/Models/.
        Custom_objects is a dictionary resolving the names of all the custom objects used during the creation of the model. 
        Returns the autoencoder, the encoder and the decoders models, assuming the two latters are respectively the second and the third layer of the AE model.
        """
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdivSum_Between0and1Constraint/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'MaxPlusDense':MaxPlusDense, 'KL_divergence_sum':custom_regularizers.KL_divergence_sum, 'Between_0_and_1':custom_constraints.Between_0_and_1}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]
        loaded_AE.nb_input_channels = loaded_AE.encoder.input_shape[-1]
        loaded_AE.nb_rows = loaded_AE.encoder.input_shape[1]
        loaded_AE.nb_columns = loaded_AE.encoder.input_shape[2]
        loaded_AE.nb_output_channels = loaded_AE.decoder.output_shape[-1]
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['rho']
        return loaded_AE        
    
    def save(self, path_to_model_directory="../Results/ShallowAE_MaxPlus/", model_name=None):
        """
        Save the model as a h5 file under the following path: path_to_model_directory/Simmple/Models/yy_mm_dd_dim'latent_dim'.h5
        model_name: String or None, if specified, it is used as a suffix to the previous name.
        """
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdivSum_Between0and1Constraint/Models/"
        strDate = d.strftime("%y_%m_%d")
        if model_name is None:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdivSum_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_Between0and1Constraint.h5"
        else:
            model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdivSum_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_Between0and1Constraint" + model_name + ".h5"
        self.autoencoder.save(model_path)

    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective}
