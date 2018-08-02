import numpy as np
from testsShallowAE import testShallowAEnoAMD
from testShallowAEwithAMD import testShallowAEwithAMD
from testKLdiv import test_KL_div
import keras

print(keras.__version__)
#dims = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 784]

#testShallowAE(latent_dimensions=dims, svm=True)

testShallowAEwithAMD(latent_dimensions=[100], svm=True)

#sparsity_weights = [0.01, 0.1, 0.5, 1, 10]
#sparsity_objectives = [0.01, 0.05, 0.1, 0.2]
#dims = [5, 10, 50, 100, 200, 500]

#test_KL_div(latent_dimension=100, sparsity_weights=sparsity_weights, sparsity_objectives=sparsity_objectives, svm=False)

testShallowAEnoAMD(latent_dimensions=[100], svm=True)

#sparsity_weights = [0.01, 0.1, 0.5, 1, 10]
#sparsity_objectives = [0.01, 0.05, 0.1, 0.2]
#dims = [5, 10, 50, 100, 200, 500]

#for d in dims:
#    test_KL_div(latent_dimension=d, sparsity_weights=sparsity_weights, sparsity_objectives=sparsity_objectives, svm=False)

