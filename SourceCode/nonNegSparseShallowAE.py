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

class Sparse_NonNeg_ShallowAE_KL_AsymDecay(ShallowAE):
    
    def __init__(self, latent_dim=100, sparsity_weight=1, sparsity_objective=0.1, decay_positive_weights=0, decay_negative_weights=1, decay_weight=1):
        """
        Create a sparse shallow AE with the custom kl divergence regularizer, enforcing weights non negativity with an asymmetric decay.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
            sparsity_objective: float between 0 and 1 - the sparsity parameter.
            decay_positive_weights: positive float - the weight decay parameter for the positive weights.
            decay_negative_weights: positive float - the weight decay parameter for the negative weights.
            decay_weight: positive float - the weight of the whole non negativity cost.
        """
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight
        self.sparsity_objective = sparsity_objective
        self.decay_positive_weights = decay_positive_weights        
        self.decay_negative_weights = decay_negative_weights
        self.decay_weight = decay_weight
        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', activity_regularizer=custom_regularizers.KL_divergence(beta=self.sparsity_weight, rho=self.sparsity_objective), kernel_regularizer=custom_regularizers.asymmetric_weight_decay(alpha=self.decay_positive_weights, beta=self.decay_negative_weights, lam=self.decay_weight))(x)
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
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdiv_AsymDecay/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'KL_divergence':custom_regularizers.KL_divergence, 'asymmetric_weight_decay':custom_regularizers.asymmetric_weight_decay}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['rho']
        loaded_AE.decay_positive_weights = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['alpha']
        loaded_AE.decay_negative_weights = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['beta']
        loaded_AE.decay_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['lam']
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/"):
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdiv_AsymDecay/Models/"
        strDate = d.strftime("%y_%m_%d")
        model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdiv_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_AsymDecay_" + str(self.decay_positive_weights) + "_" + str(self.decay_negative_weights) + "_" + str(self.decay_weight) + ".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective, 'decay_positive_weights':self.decay_positive_weights, 'decay_negative_weights':self.decay_negative_weights, 'decay_weight':self.decay_weight}


class Sparse_NonNeg_ShallowAE_KL_NonNegConstraint(ShallowAE):
    
    def __init__(self, latent_dim=100, sparsity_weight=1, sparsity_objective=0.1):
        """
        Create a sparse shallow AE with the custom kl divergence regularizer, enforcing weights non negativity with Keras NonNeg constraint.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
            sparsity_objective: float between 0 and 1 - the sparsity parameter.
        """
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight
        self.sparsity_objective = sparsity_objective
        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', activity_regularizer=custom_regularizers.KL_divergence(beta=self.sparsity_weight, rho=self.sparsity_objective), kernel_constraint=constraints.non_neg())(x)
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
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdiv_NonNegConstraint/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'KL_divergence':custom_regularizers.KL_divergence}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['rho']
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/"):
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdiv_NonNegConstraint/Models/"
        strDate = d.strftime("%y_%m_%d")
        model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdiv_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_NonNegConstraint.h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective}


class Sparse_NonNeg_ShallowAE_L1_AsymDecay(ShallowAE):
    
    def __init__(self, latent_dim=100, sparsity_weight=1, decay_positive_weights=0, decay_negative_weights=1, decay_weight=1):
        """
        Create a sparse shallow AE with the custom kl divergence regularizer, enforcing Non Negativity with asymmetric weight decay.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
            decay_positive_weights: positive float - the weight decay parameter for the positive weights.
            decay_negative_weights: positive float - the weight decay parameter for the negative weights.
            decay_weight: positive float - the weight of the whole non negativity cost.
        """
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight
        self.decay_positive_weights = decay_positive_weights        
        self.decay_negative_weights = decay_negative_weights
        self.decay_weight = decay_weight
        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', activity_regularizer=regularizers.l1(self.sparsity_weight), kernel_regularizer=custom_regularizers.asymmetric_weight_decay(alpha=self.decay_positive_weights, beta=self.decay_negative_weights, lam=self.decay_weight))(x)
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
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/L1_AsymDecay/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'asymmetric_weight_decay':custom_regularizers.asymmetric_weight_decay}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['l1']
        loaded_AE.decay_positive_weights = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['alpha']
        loaded_AE.decay_negative_weights = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['beta']
        loaded_AE.decay_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['lam']
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/"):
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/L1_AsymDecay/Models/"
        strDate = d.strftime("%y_%m_%d")
        model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + + "_L1_" + str(self.sparsity_weight) + "_Asym_Decay_" + str(self.decay_positive_weights) + "_" + str(self.decay_negative_weights) + "_" + str(self.decay_weight) + ".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'decay_positive_weights':self.decay_positive_weights, 'decay_negative_weights':self.decay_negative_weights, 'decay_weight':self.decay_weight}


class Sparse_NonNeg_ShallowAE_L1_NonNegConstraint(ShallowAE):
    
    def __init__(self, latent_dim=100, sparsity_weight=1):
        """
        Create a sparse shallow AE with the custom kl divergence regularizer.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
        """
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight
        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', activity_regularizer=regularizers.l1(self.sparsity_weight), kernel_constraint=constraints.non_neg())(x)
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
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/L1_NonNegConstraint/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=custom_objects)
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['l1']
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/"):
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/L1_NonNegConstraint/Models/"
        strDate = d.strftime("%y_%m_%d")
        model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_L1_" + str(self.sparsity_weight) + "_NonNegConstraint.h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight}


class Sparse_NonNeg_ShallowAE_KLsum_AsymDecay(ShallowAE):
    
    def __init__(self, latent_dim=100, sparsity_weight=0.1, sparsity_objective=0.1, decay_positive_weights=0, decay_negative_weights=1, decay_weight=1):
        """
        Create a sparse shallow AE with the custom kl divergence regularizer, enforcing weights non negativity with an asymmetric decay.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
            sparsity_objective: float between 0 and 1 - the sparsity parameter.
            decay_positive_weights: positive float - the weight decay parameter for the positive weights.
            decay_negative_weights: positive float - the weight decay parameter for the negative weights.
            decay_weight: positive float - the weight of the whole non negativity cost.
        """
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight
        self.sparsity_objective = sparsity_objective
        self.decay_positive_weights = decay_positive_weights        
        self.decay_negative_weights = decay_negative_weights
        self.decay_weight = decay_weight
        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', activity_regularizer=custom_regularizers.KL_divergence_sum(beta=self.sparsity_weight, rho=self.sparsity_objective), kernel_regularizer=custom_regularizers.asymmetric_weight_decay(alpha=self.decay_positive_weights, beta=self.decay_negative_weights, lam=self.decay_weight))(x)
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
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdivSum_AsymDecay/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'KL_divergence_sum':custom_regularizers.KL_divergence_sum, 'asymmetric_weight_decay':custom_regularizers.asymmetric_weight_decay}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['rho']
        loaded_AE.decay_positive_weights = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['alpha']
        loaded_AE.decay_negative_weights = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['beta']
        loaded_AE.decay_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['kernel_regularizer']['config']['lam']
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/"):
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdivSum_AsymDecay/Models/"
        strDate = d.strftime("%y_%m_%d")
        model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdivSum_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_AsymDecay_" + str(self.decay_positive_weights) + "_" + str(self.decay_negative_weights) + "_" + str(self.decay_weight) + ".h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective, 'decay_positive_weights':self.decay_positive_weights, 'decay_negative_weights':self.decay_negative_weights, 'decay_weight':self.decay_weight}


class Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint(ShallowAE):
    
    def __init__(self, latent_dim=100, sparsity_weight=0.1, sparsity_objective=0.1):
        """
        Create a sparse shallow AE with the custom kl divergence regularizer, enforcing weights non negativity with Keras NonNeg constraint.
        Arguments:
            sparsity_weight: positive float - the weight of the sparsity cost.
            sparsity_objective: float between 0 and 1 - the sparsity parameter.
        """
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight
        self.sparsity_objective = sparsity_objective
        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        x = Flatten()(input_img)
        encoded = Dense(latent_dim, activation='sigmoid', activity_regularizer=custom_regularizers.KL_divergence_sum(beta=self.sparsity_weight, rho=self.sparsity_objective), kernel_constraint=constraints.non_neg())(x)
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
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdivSum_NonNegConstraint/Models/"
        model_path = path_to_directory + model_name
        loaded_AE = cls()
        loaded_AE.autoencoder = load_model(model_path, custom_objects=dict({'KL_divergence_sum':custom_regularizers.KL_divergence_sum}, **custom_objects))
        loaded_AE.encoder = loaded_AE.autoencoder.layers[1]
        loaded_AE.decoder = loaded_AE.autoencoder.layers[2]
        loaded_AE.latent_dim = loaded_AE.encoder.output_shape[1]        
        loaded_AE.sparsity_weight = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['beta']
        loaded_AE.sparsity_objective = loaded_AE.encoder.get_config()['layers'][2]['config']['activity_regularizer']['config']['rho']
        return loaded_AE

    def save(self, path_to_model_directory="../ShallowAE/"):
        d = datetime.date.today()
        path_to_directory = path_to_model_directory + "Sparse_NonNeg/KLdivSum_NonNegConstraint/Models/"
        strDate = d.strftime("%y_%m_%d")
        model_path = path_to_directory + strDate + "_dim" + str(self.latent_dim) + "_KLdivSum_" + str(self.sparsity_weight) + "_" + str(self.sparsity_objective) + "_NonNegConstraint.h5"
        self.autoencoder.save(model_path)
    
    def get_parameters_value(self):
        return {'sparsity_weight':self.sparsity_weight, 'sparsity_objective':self.sparsity_objective}

