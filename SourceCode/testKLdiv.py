import numpy as np
import bastien_utils
from shallowAE import ShallowAE
from sparseShallowAE import SparseShallowAE_KL, SparseShallowAE_L1, SparseShallowAE_KL_sum
from nonNegSparseShallowAE import Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint
import datetime
import pandas as pd
import morphoMaths

PATH_TO_DATA = "../"

def test_KL_div(sparsity_weights = [1], sparsity_objectives = [0.1], latent_dimension=100, nb_epochs=400, 
                nb_input_channels=1, one_channel_output=True,
                AMD=False, AMD_step=1, AMD_init_step=1, svm=False, 
                path_to_dir = "../ShallowAE/"):
    x_train, ytrain, x_test, y_test = bastien_utils.load_data(PATH_TO_DATA, train=True, test=True, subsetTest=False)
    if (nb_input_channels>1):
        if AMD:
            if (nb_input_channels>2):
                x_train = morphoMaths.AMD_in_one_array(x_train[:,:,:,0], levels=nb_input_channels-2, step=AMD_step, init_step=AMD_init_step)
                x_test = morphoMaths.AMD_in_one_array(x_test[:,:,:,0], levels=nb_input_channels-2, step=AMD_step, init_step=AMD_init_step)
                path_to_dir=path_to_dir+"/SeveralChannels/WithAMD/"
            else:
                print('nb_input_channels is supposed to be greater than 2 to be used with AMD: ...setting to 1...')
                nb_input_channels=1
        else:        
            x_train = np.tile(x_train, (1,1,1,nb_input_channels))
            x_test = np.tile(x_test, (1,1,1,nb_input_channels))
            path_to_dir=path_to_dir+"/SeveralChannels/NoAMD/"
    d = datetime.date.today()
    strDims = 'dim' + str(latent_dimension) 
    strDate = d.strftime("%y_%m_%d")
    out_path = path_to_dir + "Sparse/KL_div_sum/TestOutputs/" + strDate
    nb_sparsity_weights = len(sparsity_weights)
    nb_sparsity_objectives = len(sparsity_objectives)
    train_rec_errors = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    test_rec_errors = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    train_kl_loss = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    test_kl_loss = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    sparsity_train = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    sparsity_test = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    max_approx_error_toOriginal_train = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    max_approx_error_toOriginal_test = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    max_approx_error_toRec_train = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    max_approx_error_toRec_test = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    np.save(out_path +'_sparsityWeights_' + strDims, sparsity_weights)
    np.save(out_path +'_sparsityObjectives_' + strDims, sparsity_objectives)
    if svm:
        SVM_classification_accuracy = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    for idx1, sp_w in enumerate(sparsity_weights):
        for idx2, sp_o in enumerate(sparsity_objectives):
            shAE = SparseShallowAE_KL_sum(latent_dim=latent_dimension, sparsity_weight=sp_w, sparsity_objective=sp_o, nb_input_channels=nb_input_channels, one_channel_output=one_channel_output)
            shAE.train(x_train, nb_epochs=nb_epochs, X_val=x_test, verbose=2)
            shAE.save(path_to_model_directory=path_to_dir)
            train_rec_errors[idx1, idx2] =shAE.reconstruction_error(x_train)
            test_rec_errors[idx1, idx2] = shAE.reconstruction_error(x_test)
            train_kl_loss[idx1, idx2] = shAE.total_loss(x_train) - train_rec_errors[idx1, idx2]
            test_kl_loss[idx1, idx2] = shAE.total_loss(x_test) - test_rec_errors[idx1, idx2] 
            sparsity_train[idx1, idx2] = shAE.sparsity_measure(x_train)
            sparsity_test[idx1, idx2] = shAE.sparsity_measure(x_test)
            max_approx_train = shAE.max_approximation_error(x_train, morphoMaths.dilatation, apply_to_bias=True, SE_scale=1)
            max_approx_error_toOriginal_train[idx1, idx2] = max_approx_train[0]
            max_approx_error_toRec_train[idx1, idx2] = max_approx_train[1]
            max_approx_test = shAE.max_approximation_error(x_test, morphoMaths.dilatation, apply_to_bias=True, SE_scale=1)
            max_approx_error_toOriginal_test[idx1, idx2] = max_approx_test[0]
            max_approx_error_toRec_test[idx1, idx2] = max_approx_test[1]
            np.save(out_path +'_training_errors_' + strDims, train_rec_errors)
            np.save(out_path +'_test_errors_' + strDims, test_rec_errors)
            np.save(out_path +'_training_kl_loss_' + strDims, train_kl_loss)
            np.save(out_path +'_test_kl_loss_' + strDims, test_kl_loss)
            np.save(out_path +'_training_sparsity_' + strDims, sparsity_train)
            np.save(out_path +'_test_sparsity_' + strDims, sparsity_test)
            np.save(out_path +'_training_max_approx_error_toOriginal_dilatation_' + strDims, max_approx_error_toOriginal_train)
            np.save(out_path +'_test_max_approx_error_toOriginal_dilation_' + strDims, max_approx_error_toOriginal_test)
            np.save(out_path +'_training_max_approx_error_toRec_dilatation_' + strDims, max_approx_error_toRec_train)
            np.save(out_path +'_test_max_approx_error_toRec_dilation_' + strDims, max_approx_error_toRec_test)
            if svm:
                SVM_classification_accuracy[idx1, idx2] = shAE.best_SVM_classification_score(x_test, y_test, nb_values_C=10, nb_values_gamma=10)[0]
                np.save(out_path +'_svm_acc_' + strDims, SVM_classification_accuracy)
    if svm:
        results = pd.DataFrame(data={'sparsity_weight':np.repeat(sparsity_weights,len(sparsity_objectives)), 'sparsity_objective':np.tile(sparsity_objectives, len(sparsity_weights)),
                                'training_error':train_rec_errors.flatten(), 'test_error':test_rec_errors.flatten(),
                                'training_kl_loss':train_kl_loss.flatten(), 'test_kl_loss':test_kl_loss.flatten(),
                                'training_sparsity':sparsity_train.flatten(), 'test_sparsity':sparsity_test.flatten(),
                                'training_max_approx_error_toOriginal_dilatation':max_approx_error_toOriginal_train.flatten(),
                                'test_max_approx_error_toOriginal_dilatation':max_approx_error_toOriginal_test.flatten(),
                                'training_max_approx_error_toRec_dilatation':max_approx_error_toRec_train.flatten(),
                                'test_max_approx_error_toRec_dilatation':max_approx_error_toRec_test.flatten(),
                                'SVM_classification_score':SVM_classification_accuracy.flatten()})
    else:
        results = pd.DataFrame(data={'sparsity_weight':np.repeat(sparsity_weights,len(sparsity_objectives)), 'sparsity_objective':np.tile(sparsity_objectives, len(sparsity_weights)),
                                'training_error':train_rec_errors.flatten(), 'test_error':test_rec_errors.flatten(),
                                'training_kl_loss':train_kl_loss.flatten(), 'test_kl_loss':test_kl_loss.flatten(),
                                'training_sparsity':sparsity_train.flatten(), 'test_sparsity':sparsity_test.flatten(),
                                'training_max_approx_error_toOriginal_dilatation':max_approx_error_toOriginal_train.flatten(),
                                'test_max_approx_error_toOriginal_dilatation':max_approx_error_toOriginal_test.flatten(),
                                'training_max_approx_error_toRec_dilatation':max_approx_error_toRec_train.flatten(),
                                'test_max_approx_error_toRec_dilatation':max_approx_error_toRec_test.flatten()})
    results.to_csv(out_path+'results')