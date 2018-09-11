####Computes and save the svm classification scores 
# and the max_approximation to dilatation with and without dilating the bias 
# for all the models in a given directory

from os import listdir
import numpy as np
from nonNegSparseShallowAE import Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint
from AsymAE_infoGAN.nonNegSparseAsymAEinfoGAN import Sparse_NonNeg_AsymAEinfoGAN_KLsum_NonNegConstraint
import morphoMaths
import bastien_utils
PATH_TO_DATA = "../"
def recomputeMaxApprox():
    """
    Creating a super class for all auto-encoders would enable not to worry about the model being of a shallowAE or a AsymAE
    Note that a shallowAE eventhough 
    """
    path_to_model_dir = '../Results/AsymAE_infoGAN/Sparse_NonNeg/KLdivSum_NonNegConstraint/Models/'
    path_to_output_dir = '../Results/AsymAE_infoGAN/Sparse_NonNeg/KLdivSum_NonNegConstraint/TestOutputs/'
    out_path=path_to_output_dir+'18_09_04'
    strDims = 'dim100'
    x_train, _, x_test, _ = bastien_utils.load_data(PATH_TO_DATA, train=True, test=True, subsetTest=False)
    models = listdir(path_to_model_dir)
    max_approx_error_toOriginal_train = np.zeros((8,4))
    max_approx_error_toOriginal_test = np.zeros((8,4))
    max_approx_error_toRec_train = np.zeros((8,4))
    max_approx_error_toRec_test = np.zeros((8,4))
    objectives = [0.01, 0.05, 0.1, 0.2]
    weights = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    for m in models:
        print('Model= ', m)
        sp_o = m.split('_')[-2]
        sp_w = m.split('_')[-3]
        print('sp_w: ', sp_w, ' - sp_o: ', sp_o)
        idx_w = weights.index(float(sp_w))
        idx_o = objectives.index(float(sp_o))
        print('idx_w: ', idx_w, ' - idx_o: ', idx_o, '\n')
        shAE = Sparse_NonNeg_AsymAEinfoGAN_KLsum_NonNegConstraint.load(m)
        max_approx_train = shAE.max_approximation_error(x_train, morphoMaths.dilatation, original_images=x_train, apply_to_bias=False, SE_scale=1)
        max_approx_error_toOriginal_train[idx_w, idx_o] = max_approx_train[0]
        max_approx_error_toRec_train[idx_w, idx_o] = max_approx_train[1]
        max_approx_test = shAE.max_approximation_error(x_test, morphoMaths.dilatation, original_images=x_test, apply_to_bias=False, SE_scale=1)
        max_approx_error_toOriginal_test[idx_w, idx_o] = max_approx_test[0]
        max_approx_error_toRec_test[idx_w, idx_o] = max_approx_test[1]
    np.save(out_path +'_training_max_approx_error_toOriginal_dilatation_' + strDims, max_approx_error_toOriginal_train)
    np.save(out_path +'_test_max_approx_error_toOriginal_dilation_' + strDims, max_approx_error_toOriginal_test)
    np.save(out_path +'_training_max_approx_error_toRec_dilatation_' + strDims, max_approx_error_toRec_train)
    np.save(out_path +'_test_max_approx_error_toRec_dilation_' + strDims, max_approx_error_toRec_test)





