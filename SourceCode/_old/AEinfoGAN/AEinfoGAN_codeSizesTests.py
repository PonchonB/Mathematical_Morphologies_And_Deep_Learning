from __future__ import print_function

import numpy as np
import sys
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm
from keras import backend as K

sys.path.append('/home/ponchon/Stage/fashion_mnist')

from utils import mnist_reader

import AEinfoGAN

#if len(sys.argv) < 2:
#    print >> sys.stderr, "Usage: %s <code_size>"  % (sys.argv[0])
#    sys.exit(1)

#latent_dim = int(sys.argv[1])

img_rows, img_cols, img_chns = 28, 28, 1

if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)

x_train, y_train = mnist_reader.load_mnist('../fashion_mnist/data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('../fashion_mnist/data/fashion', kind='t10k')


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


nb_models = 2 
min_dim = 5
dim_step = 5
max_dim = (nb_models-1)*dim_step + min_dim

sparse = True
save=True

d = datetime.date.today()
strDate = d.strftime("%y_%m_%d")

if sparse:
    dir = "../AEinfoGAN/Sparse/TrainingOutputs/new/" + strDate
else:
    dir = "../AEinfoGAN/Simple/TrainingOutputs/new/" + strDate

strDims = str(min_dim) + '_' + str(max_dim) + '_' + str(nb_models)


dimensions = np.array([min_dim + dim_step*k for k in range(nb_models)])
training_times = np.zeros(nb_models)
test_errors = np.zeros(nb_models)
training_errors = np.zeros(nb_models)

np.save(dir + '_dimensions_' + strDims, dimensions)

for idx, k in enumerate(dimensions):
    dim = int(k)
    training_time, training_error, test_error = AEinfoGAN.createAndTrain(x_train, y_train, x_test, y_test, dim, strDate, sparse=sparse, save=save)
    training_times[idx] = training_time
    training_errors[idx] = training_error
    test_errors[idx] = test_error
    np.save(dir +'_training_times_' + strDims, training_times)
    np.save(dir +'_training_errors_' + strDims, training_errors)
    np.save( dir +'_test_errors_' + strDims, test_errors)

np.save(dir +'_training_times_' + strDims, training_times)
np.save(dir +'_training_errors_' + strDims, training_errors)
np.save(dir +'_test_errors_' + strDims, test_errors)




