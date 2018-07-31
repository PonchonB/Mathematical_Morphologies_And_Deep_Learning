import numpy as np
from testsShallowAE import testShallowAE
from testKLdiv import test_KL_div
from testShallowAEwithAMD import testShallowAEwithAMD

#dims = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 784]

#testShallowAE(latent_dimensions=dims, svm=True)

#testShallowAEwithAMD(latent_dimensions=dims, svm=True)

sparsity_weights = [0.01, 0.1, 0.5, 1, 10]
sparsity_objectives = [0.01, 0.05, 0.1, 0.2]

test_KL_div(sparsity_weights=sparsity_weights, sparsity_objectives=sparsity_objectives, svm=True)

