from __future__ import print_function

import numpy as np
import time

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Conv2DTranspose, LeakyReLU, BatchNormalization, Activation
from keras.models import Model
from keras import losses, regularizers
from keras import metrics

import custom_regularizers

import h5py
from keras.regularizers import l2, Regularizer
import keras.backend as K

def kl_divergence(output):
    s_hat = K.mean(output, 0)
    s = 0.01
    s_hat += 10 ** -5
    val = s * K.log(s/s_hat) + (1 - s) * K.log((1 - s)/(1 - s_hat))
    return val


def createAndTrain(x_train, y_train, x_test, y_test, latent_dim, strDate, sparse=False, save=False):

	print('\n\n **************** Code size: ', latent_dim, ' **************** \n')


	input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
	x = Flatten()(input_img)

	if sparse:
		encoded = Dense(latent_dim, activation='sigmoid',  activity_regularizer=custom_regularizers.KL_divergence())(x)
	else:
		encoded = Dense(latent_dim, activation='sigmoid')(x)

	encoder = Model(input_img, encoded, name='encoder')
	encoder.summary()

	encoded_img = Input(shape=(latent_dim,))  # adapt this if using `channels_first` image data format

	x = Dense(28*28)(encoded_img)
	x = LeakyReLU(alpha=0.1)(x)
	decoded = Reshape((28,28,1))(x)

	decoder = Model(encoded_img, decoded, name='decoder')
	
	encoded = encoder(input_img)
	decoded = decoder(encoded)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
	
	encoded = encoder(input_img)
	decoded = decoder(encoded)
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	t0 = time.time()

	autoencoder.fit(x_train, x_train,
		        epochs=200,
			verbose=2,
		        batch_size=128,
		        shuffle=True,
		        validation_data=(x_test, x_test),
		        )

	t1 = time.time()

	if ((latent_dim % 50==0) or (save==True)):
		model_path = dir + '_AEinfoGAN_' + str(latent_dim) +'.h5'
		autoencoder.save(model_path)
	training_time = t1-t0
	training_error = autoencoder.evaluate(x_train, x_train, verbose=0)
	test_error = autoencoder.evaluate(x_test, x_test, verbose=0)

	return training_time, training_error, test_error

