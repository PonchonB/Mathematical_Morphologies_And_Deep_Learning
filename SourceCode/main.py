import numpy as np
from testsShallowAE import testShallowAEnoAMD
from testShallowAEwithAMD import testShallowAEwithAMD
from nonNegShallowAE import NonNegShallowAE_NonNegConstraint
from sparseShallowAE import SparseShallowAE_KL_sum
from nonNegSparseShallowAE import Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint
from testKLdiv import test_KL_div
from testAE import testDims
import keras

print("Keras version: ", keras.__version__)
#dims = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 784]

#testShallowAE(latent_dimensions=dims, svm=True)

#testShallowAEwithAMD(latent_dimensions=[100], svm=True)

sparsity_weights = [0.01, 0.1, 0.5, 1, 3]
sparsity_objectives = [0.01, 0.05, 0.1, 0.2]
#dims = [5, 10, 50, 100, 200, 500]

#test_KL_div(latent_dimension=100, sparsity_weights=sparsity_weights, sparsity_objectives=sparsity_objectives, svm=False)

#test_KL_div(latent_dimension=100, sparsity_weights=sparsity_weights, sparsity_objectives=sparsity_objectives, svm=False
#            , nb_input_channels=6, one_channel_output=True, AMD=True)


#testShallowAEnoAMD(latent_dimensions=[100], svm=True)

#testDims(ShallowAE_class=NonNegShallowAE_NonNegConstraint, latent_dimensions=[100], svm=True, nonNeg=True)

#testDims(svm=True)

#testDims(ShallowAE_class=SparseShallowAE_KL_sum, svm=True, sparsity_weight=, sparsity_objective=)

#dims = [5, 10, 50, 100, 200, 500]

#for d in dims:
#    test_KL_div(latent_dimension=d, sparsity_weights=sparsity_weights, sparsity_objectives=sparsity_objectives, svm=False)

#testDims(svm=True, nb_input_channels=5, AMD=True, add_original_images=False)

###18_08_07
#testDims(ShallowAE_class=NonNegShallowAE_NonNegConstraint, nb_epochs=500, svm=True, nb_input_channels=6, AMD=True, add_original_images=False)

###18_08_08
#test_KL_div(ShallowAE_class=SparseShallowAE_KL_sum, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,
#            latent_dimension=100, nb_input_channels=6, one_channel_output=True, AMD=True, add_original_images=False)

###18_08_09
test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint, nb_epochs=500, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,
            latent_dimension=100, nb_input_channels=6, one_channel_output=True, AMD=True, add_original_images=False)