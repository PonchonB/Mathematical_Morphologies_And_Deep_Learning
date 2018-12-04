import numpy as np
from nonNegShallowAE import NonNegShallowAE_NonNegConstraint
from sparseShallowAE import SparseShallowAE_KL_sum
from nonNegSparseShallowAE import Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint, Sparse_NonNeg_ShallowAE_KLsum_AsymDecay, Sparse_NonNeg_ShallowAE_Hoyer_NonNegConstraint
from shallowAE import ShallowAE
from testKLdiv import test_KL_div
from testSparsityHoyer import test_Hoyer_sparsity
from testAE import testDims
import keras
from AsymAE_infoGAN.testAsymAE import testDims_AsymAE
from AsymAE_infoGAN.AsymAE_testKLdiv import test_KL_div_Asym_AE
from AsymAE_infoGAN.nonNegSparseAsymAEinfoGAN import Sparse_NonNeg_AsymAEinfoGAN_KLsum_NonNegConstraint, Sparse_NonNeg_AsymAEinfoGAN_Hoyer_NonNegConstraint

from AsymAE_infoGAN.AsymAE_infoGAN import AsymAEinfoGAN
from AsymAE_infoGAN.AsymAE_testHoyer import test_Hoyer_Asym_AE

from MaxPlusDecoder.ShallowAE_maxPlusDecoder.shallowAE_maxplus_NonNeg import NonNeg_ShallowAE_MaxPlus_Between0and1Constraint
from MaxPlusDecoder.ShallowAE_maxPlusDecoder.shallowAE_maxplus_sparseNonNeg import Sparse_NonNeg_ShallowAE_MaxPlus_KLsum_Between0and1Constraint
from MaxPlusDecoder.ShallowAE_maxPlusDecoder.shallowAE_maxplus import ShallowAE_MaxPlus


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
#test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint, nb_epochs=500, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,
#            latent_dimension=100, nb_input_channels=6, one_channel_output=True, AMD=True, add_original_images=False)

###18_08_09
#test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint, nb_epochs=500, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,
#            latent_dimension=100, nb_input_channels=7, one_channel_output=True, AMD=True, add_original_images=True)

###18_08_11
#testDims(ShallowAE_class=ShallowAE, nb_epochs=500, nb_input_channels=1, svm=True)

###18_08_12
#testDims(ShallowAE_class=NonNegShallowAE_NonNegConstraint, nb_epochs=500, nb_input_channels=1, svm=True)

####18_08_14
#test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint, nb_epochs=500, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,
#            latent_dimension=100, nb_input_channels=1, one_channel_output=True)

sparsity_weights = [0.0001, 0.0005, 0.001, 0.005]
sparsity_objectives = [0.01, 0.05, 0.1, 0.2]

###18_08_16
#test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint, nb_epochs=500, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,#test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_KLsum_AsymDecay, nb_epochs=500, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,
#            latent_dimension=100, nb_input_channels=1, one_channel_output=True)

sparsity_weights = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
sparsity_objectives = [0.01, 0.05, 0.1, 0.2]

###18_08_16
#test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_KLsum_AsymDecay, nb_epochs=500, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,
#            latent_dimension=100, nb_input_channels=1, one_channel_output=True)

###18_08_17
###### Modifying the Asym decay class so that the asym decay loss apply both to the encoder and the decoder
###### These parameters are those from the paper [Hosseini-Asl, Zurada, Nasraoui 2016]
#testDims(ShallowAE_class=Sparse_NonNeg_ShallowAE_KLsum_AsymDecay, nb_epochs=500, nb_input_channels=1, one_channel_output=True,
#        svm=True, sparsity_weight=3, sparsity_objective=0.05, decay_weight=0.003)

###18_08_17
#testDims(nb_epochs=500, nb_input_channels=6, one_channel_output=True,
#        svm=True, AMD=True, add_original_images=True)

###18_08_21
####### Modifying the testDims so that we do not rescale residutes anymore
#testDims(nb_epochs=500, nb_input_channels=6, one_channel_output=True,
#        svm=True, AMD=True, add_original_images=True)

####Adding back the residutes rescaling

###18_08_21
#test_KL_div(ShallowAE_class=SparseShallowAE_KL_sum, nb_epochs=500, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,
#            latent_dimension=100, nb_input_channels=1, one_channel_output=True)

###18_08_22
#testDims(ShallowAE_class=NonNegShallowAE_NonNegConstraint, nb_epochs=500, nb_input_channels=7, one_channel_output=True,
#        svm=True, AMD=True, add_original_images=True)

###18_08_22
#test_KL_div(ShallowAE_class=SparseShallowAE_KL_sum, nb_epochs=500, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,
#            latent_dimension=100, nb_input_channels=7, one_channel_output=True, AMD=True, add_original_images=True)

###18_08_30
### testing simple AE with AEinfoGAN as encoder
#testDims_AsymAE(svm=True)

###18_09_04
### testing AE with AEinfoGAN and KLdivSum/NonNegConstraint
### Note that the repository hierarchy has been changed (putting all Models/TestOutputs in a Results directory)
#test_KL_div_Asym_AE(AsymAE_class=Sparse_NonNeg_AsymAEinfoGAN_KLsum_NonNegConstraint, sparsity_weights = sparsity_weights, sparsity_objectives = sparsity_objectives, latent_dimension=100, nb_epochs=500, 
#                nb_input_channels=1, one_channel_output=True, add_original_images=True,
#                AMD=False, AMD_step=1, AMD_init_step=1, svm=False, 
#                path_to_dir = "../Results/AsymAE_infoGAN/")

###18_09_06
### testing AE with AEinfoGAN, Simple and AMD
#testDims_AsymAE(AsymAE_class=AsymAEinfoGAN, latent_dimensions=[100], nb_epochs=500, nb_input_channels=6, one_channel_output=True,
#            AMD=True, AMD_step=1, AMD_init_step=1, add_original_images=False,
#            svm=True)

###18_09_07
### testing AE with AEinfoGAN, Simple, With AMD and Original images
#testDims_AsymAE(AsymAE_class=AsymAEinfoGAN, latent_dimensions=[100], nb_epochs=500, nb_input_channels=7, one_channel_output=True,
#            AMD=True, AMD_step=1, AMD_init_step=1, add_original_images=True,
#            svm=True)

###18_09_08
###test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint, nb_epochs=500, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,
###            latent_dimension=100, nb_input_channels=1, one_channel_output=True, svm=False)

###18_09_08
#test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint, nb_epochs=500, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,
#            latent_dimension=100, nb_input_channels=6, one_channel_output=True, svm=False, PADO=True, AMD_step=1, AMD_init_step=1, add_original_images=False)

###18_09_10
#testDims(ShallowAE_class=ShallowAE, latent_dimensions=[100], nb_epochs=500, nb_input_channels=6, one_channel_output=True,
#            AMD=False, PADO=True, AMD_step=1, AMD_init_step=1, add_original_images=False,
#            svm=True)

###18_09_10
###Sparse_NonNeg with PADO and originals
#test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint, nb_epochs=500, sparsity_objectives=sparsity_objectives, sparsity_weights=sparsity_weights,
#            latent_dimension=100, nb_input_channels=7, one_channel_output=True, svm=False, PADO=True, AMD_step=1, AMD_init_step=1, add_original_images=True)

###18_09_11
###Simple shallowAE with PADO and originals
#testDims(ShallowAE_class=ShallowAE, latent_dimensions=[100], nb_epochs=500, nb_input_channels=7, one_channel_output=True,
#            AMD=False, PADO=True, AMD_step=1, AMD_init_step=1, add_original_images=True,
#            svm=False)

##### Correcting the max approximation error computations (no longer applying the dilatation to bias) in the four test functions


###18_09_11
### AsymAE Sparse NonNeg with PADO without originals
#test_KL_div_Asym_AE(AsymAE_class=Sparse_NonNeg_AsymAEinfoGAN_KLsum_NonNegConstraint, sparsity_weights = sparsity_weights, sparsity_objectives = sparsity_objectives, latent_dimension=100, nb_epochs=500,
#                nb_input_channels=6, one_channel_output=True, add_original_images=False,
#                AMD=False, PADO=True, AMD_step=1, AMD_init_step=1, svm=False, 
#                path_to_dir = "../Results/AsymAE_infoGAN/")

###18_09_12
### testing asymAE with AEinfoGAN, Simple, With PADO and noOriginal images
#testDims_AsymAE(AsymAE_class=AsymAEinfoGAN, latent_dimensions=[100], nb_epochs=500, nb_input_channels=6, one_channel_output=True,
#            AMD=False, PADO=True, AMD_step=1, AMD_init_step=1, add_original_images=False,
#            svm=False)

###18_09_24
#testDims_AsymAE(AsymAE_class=AsymAEinfoGAN, latent_dimensions=[100], nb_epochs=500, nb_input_channels=7, one_channel_output=True,
#            AMD=False, PADO=True, AMD_step=1, AMD_init_step=1, add_original_images=True,
#            svm=False)

###18_09_26
#test_KL_div_Asym_AE(AsymAE_class=Sparse_NonNeg_AsymAEinfoGAN_KLsum_NonNegConstraint, sparsity_weights = sparsity_weights, sparsity_objectives = sparsity_objectives, latent_dimension=100, nb_epochs=500,
#                nb_input_channels=7, one_channel_output=True, add_original_images=True,
#                AMD=False, PADO=True, AMD_step=1, AMD_init_step=1, svm=False, 
#                path_to_dir = "../Results/AsymAE_infoGAN/")

###18_10_16
###Asym AE with Sparse and NonNeg Constraint with AMD and original images 
#test_KL_div_Asym_AE(AsymAE_class=Sparse_NonNeg_AsymAEinfoGAN_KLsum_NonNegConstraint, sparsity_weights = sparsity_weights, sparsity_objectives = sparsity_objectives, latent_dimension=100, nb_epochs=500,
#                nb_input_channels=7, one_channel_output=True, add_original_images=True,
#                AMD=True, PADO=False, AMD_step=1, AMD_init_step=1, svm=False,
#                path_to_dir = "../Results/AsymAE_infoGAN/")

###18_10_25
###Asym AE with Sparse and NonNeg Constraint with AMD without original images
#test_KL_div_Asym_AE(AsymAE_class=Sparse_NonNeg_AsymAEinfoGAN_KLsum_NonNegConstraint, sparsity_weights = sparsity_weights, sparsity_objectives = sparsity_objectives, latent_dimension=100, nb_epochs=500,
#                nb_input_channels=6, one_channel_output=True, add_original_images=False,
#                AMD=True, PADO=False, AMD_step=1, AMD_init_step=1, svm=False,
#                path_to_dir = "../Results/AsymAE_infoGAN/")

###18_10_29
###ShallowAE with Sparsity Hoyer regularizer for the constraint
#test_Hoyer_sparsity(ShallowAE_class=Sparse_NonNeg_ShallowAE_Hoyer_NonNegConstraint, sparsity_weights = [0.1, 0.5, 1, 5], sparsity_objectives = [0.6], latent_dimension=100, nb_epochs=500, 
#                nb_input_channels=1, one_channel_output=True, add_original_images=True,
#                AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, svm=False, 
#                path_to_dir = "../Results/ShallowAE/")

###18_10_29
###ShallowAE with Sparsity Hoyer regularizer for the constraint
#test_Hoyer_sparsity(ShallowAE_class=Sparse_NonNeg_ShallowAE_Hoyer_NonNegConstraint, sparsity_weights = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], sparsity_objectives = [0.6], latent_dimension=100, nb_epochs=500, 
#                nb_input_channels=1, one_channel_output=True, add_original_images=True,
#                AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, svm=False, 
#                path_to_dir = "../Results/ShallowAE/")

###18_10_30
###AsymAE_infoGAN with Hoyer sparsity regularizer
#test_Hoyer_Asym_AE(AsymAE_class=Sparse_NonNeg_AsymAEinfoGAN_Hoyer_NonNegConstraint, sparsity_weights = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], sparsity_objectives = [0.6], latent_dimension=100, nb_epochs=500, 
#                nb_input_channels=1, one_channel_output=True, add_original_images=True,
#                AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, svm=False, 
#                path_to_dir = "../Results/AsymAE_infoGAN/")


##########Modification:in the path given to the function and where the results are stored

###18_11_06
###AsymAE_infoGAN with KLdiv sparsity regularizer and Non-Negative Contstrain on MNIST dataset
#test_KL_div_Asym_AE(AsymAE_class=Sparse_NonNeg_AsymAEinfoGAN_KLsum_NonNegConstraint, sparsity_weights = sparsity_weights, sparsity_objectives = sparsity_objectives, latent_dimension=100, nb_epochs=500,
#                nb_input_channels=1, one_channel_output=True, add_original_images=False,
#                AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, svm=False,
#                path_to_dir = "../Results", dataset_fashion_MNIST=False)

###18_11_12
####AsymAE_infoGAN with KLdiv sparsity regularizer and NonNegConstraint on MNIST dataset REGULARIZED BETWEEN 0 AND 1 like fashionMNIST
#test_KL_div_Asym_AE(AsymAE_class=Sparse_NonNeg_AsymAEinfoGAN_KLsum_NonNegConstraint, sparsity_weights = sparsity_weights, sparsity_objectives = sparsity_objectives, latent_dimension=100, nb_epochs=500,
#                nb_input_channels=1, one_channel_output=True, add_original_images=False,
#                AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, svm=False,
#                path_to_dir = "../Results", dataset_fashion_MNIST=False)

###18_11_24
####ShallowAE with MaxPlus Decoder and Between 0 and 1 constraint
#testDims(ShallowAE_class=NonNeg_ShallowAE_MaxPlus_Between0and1Constraint, latent_dimensions=[100], nb_epochs=100, nb_input_channels=1, one_channel_output=True,
#            AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, add_original_images=True,
#            svm=False, path_to_dir = "../Results/ShallowAE_MaxPlus/")

###18_11_24
####ShallowAE with MaxPlus Decoder and Sparse (KLdivSum)/Between0and1 constraints
#test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_MaxPlus_KLsum_Between0and1Constraint, sparsity_weights = sparsity_weights, sparsity_objectives = sparsity_objectives, latent_dimension=100, nb_epochs=200, 
#                nb_input_channels=1, one_channel_output=True, add_original_images=True,
#                AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, svm=False, 
#                path_to_dir = "../Results/ShallowAE_MaxPlus/")

####18_11_26
####ShallowAE with MaxPlus Decoder and Between 0 and 1 constraint - 500 epochs
#testDims(ShallowAE_class=NonNeg_ShallowAE_MaxPlus_Between0and1Constraint, latent_dimensions=[100], nb_epochs=500, nb_input_channels=1, one_channel_output=True,
#            AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, add_original_images=True,
#            svm=False, path_to_dir = "../Results/ShallowAE_MaxPlus/")

####18_11_26
####ShallowAE with MaxPlus Decoder - No Constraints - 500 epochs
#testDims(ShallowAE_class=ShallowAE_MaxPlus, latent_dimensions=[100], nb_epochs=500, nb_input_channels=1, one_channel_output=True,
#            AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, add_original_images=True,
#            svm=False, path_to_dir = "../Results/ShallowAE_MaxPlus/")

####18_11_26
####ShallowAE with MaxPlus Decoder and Sparse (KLdivSum)/Between0and1 constraints - 500 epochs
#test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_MaxPlus_KLsum_Between0and1Constraint, sparsity_weights = sparsity_weights, sparsity_objectives = sparsity_objectives, latent_dimension=100, nb_epochs=500, 
#                nb_input_channels=1, one_channel_output=True, add_original_images=True,
#                AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, svm=False, 
#                path_to_dir = "../Results/ShallowAE_MaxPlus/")

##### ADDITION OF A  DROPOUT LAYER BEFORE THE MAXPLUS LAYER (with rate dropping 0.2)

####18_11_30
####ShallowAE with MaxPlus Decoder - No Constraints - 500 epochs
#testDims(ShallowAE_class=ShallowAE_MaxPlus, latent_dimensions=[100], nb_epochs=500, nb_input_channels=1, one_channel_output=True,
#            AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, add_original_images=True,
#            svm=False, path_to_dir = "../Results/ShallowAE_MaxPlus/")

####18_11_30
####ShallowAE with MaxPlus Decoder and Between 0 and 1 constraint - 500 epochs
#testDims(ShallowAE_class=NonNeg_ShallowAE_MaxPlus_Between0and1Constraint, latent_dimensions=[100], nb_epochs=500, nb_input_channels=1, one_channel_output=True,
#            AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, add_original_images=True,
#            svm=False, path_to_dir = "../Results/ShallowAE_MaxPlus/")

####18_11_30
####ShallowAE with MaxPlus Decoder and Sparse (KLdivSum)/Between0and1 constraints - 500 epochs
test_KL_div(ShallowAE_class=Sparse_NonNeg_ShallowAE_MaxPlus_KLsum_Between0and1Constraint, sparsity_weights = sparsity_weights, sparsity_objectives = sparsity_objectives, latent_dimension=100, nb_epochs=500, 
                nb_input_channels=1, one_channel_output=True, add_original_images=True,
                AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, svm=False, 
                path_to_dir = "../Results/ShallowAE_MaxPlus/")
