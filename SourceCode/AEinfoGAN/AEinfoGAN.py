from __future__ import print_function

import numpy as np
import time

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Conv2DTranspose, LeakyReLU, BatchNormalization, Activation
from keras.models import Model
from keras import losses, regularizers
from keras import metrics

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
	x = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(input_img)
	x = LeakyReLU(alpha=0.1)(x)
	x = Conv2D(128, (4, 4), strides=(2,2), padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)
	x = Flatten()(x)
	x = Dense(1024)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)

	if sparse:
		dir = '../AEinfoGAN/Sparse/Models/new/' + strDate
		#encoded = Dense(latent_dim, activity_regularizer=regularizers.l1(10e-5))(x)
		encoded = Dense(latent_dim, activation='sigmoid', activity_regularizer=kl_divergence)(x)
	else:
		encoded = Dense(latent_dim, activation='sigmoid')(x)
		dir = '../AEinfoGAN/Simple/Models/new/' + strDate

	encoder = Model(input_img, encoded, name='encoder')

	encoded_img = Input(shape=(latent_dim,))  # adapt this if using `channels_first` image data format
	x = Dense(1024)(encoded_img)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)
#	x = Activation('relu')(x)
	x = Dense(6272)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)
#	x = Activation('relu')(x)
	x = Reshape((7,7,128))(x)
	x = Conv2DTranspose(64, (4, 4), strides=(2,2), padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)
#	x = Activation('relu')(x)
	decoded = Conv2DTranspose(1, (4, 4), strides=(2,2), padding='same', activation='sigmoid')(x)

	decoder = Model(encoded_img, decoded, name='decoder')

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

