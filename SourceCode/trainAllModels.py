import numpy as np
import bastien_utils
from shallowAE import ShallowAE
from sparseShallowAE import SparseShallowAE_KL, SparseShallowAE_L1, SparseShallowAE_KL_sum
from nonNegShallowAE import NonNegShallowAE_Asymmetric_decay, NonNegShallowAE_NonNegConstraint
from nonNegSparseShallowAE import Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint
import datetime
import morphoMaths


PATH_TO_MODELS_DIR = "../ShallowAE/"
PATH_TO_DATA = "../"
x_train, ytrain, x_test, y_test = bastien_utils.load_data(PATH_TO_DATA, train=True, test=True, subsetTest=False)
d = datetime.date.today()
dims = [5, 10, 100, 200, 500]
sparsity_weights = [0.01, 0.1, 0.5, 1, 10]
sparsity_objectives = [0.01, 0.05, 0.1, 0.2]

for d in dims:
    shAE = ShallowAE(latent_dim=d)
    shAE.train(x_train, nb_epochs=200, X_val=x_test, verbose=2)
    shAE.save()

for d in dims:
    test_KL_div(sparsity_weights=sparsity_weights, sparsity_objectives=sparsity_objectives, svm=False)


for d in dims:
    shAE = NonNegShallowAE_NonNegConstraint()
    shAE.train(x_train, nb_epochs=200, X_val=x_test, verbose=2)
    shAE.save()

for d in dims:
    for sp_w in sparsity_weights:
        for sp_o in sparsity_objectives:
            shAE = Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint(latent_dim=d, sparsity_weight=sp_w, sparsity_objective=sp_o)
            shAE.train(x_train, nb_epochs=200, X_val=x_test, verbose=2)
            shAE.save()

x_train_input = np.tile(x_train, (1,1,1,6))
x_test_input = np.tile(x_test, (1,1,1,6))

for d in dims:
    shAE = ShallowAE(latent_dim=d, nb_input_channels=6, one_channel_output=True)
    shAE.train(x_train_input, X_train_expected_output=x_train, nb_epochs=200, X_val=(x_test_input, x_test), verbose=2)
    shAE.save()

x_train_input = morphoMaths.AMD_in_one_array(x_train[:,:,:,0])
x_test_input = morphoMaths.AMD_in_one_array(x_test[:,:,:,0])

for d in dims:
    shAE = ShallowAE(latent_dim=d, nb_input_channels=6, one_channel_output=True)
    shAE.train(x_train_input, X_train_expected_output=x_train, nb_epochs=200, X_val=(x_test_input, x_test), verbose=2)
    shAE.save()