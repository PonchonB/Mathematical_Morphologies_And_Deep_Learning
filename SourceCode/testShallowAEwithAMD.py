import numpy as np
import bastien_utils
from shallowAE import ShallowAE
from sparseShallowAE import SparseShallowAE_KL, SparseShallowAE_L1
import datetime
import morphoMaths

PATH_TO_MODELS_DIR = "../ShallowAE/"
PATH_TO_DATA = "../"

    data = bastien_utils.load_data(PATH_TO_DATA, train=True, test=True, subsetTest=False)
    x_train, _, x_test, y_test = data
    x_train = morphoMaths.AMD_in_one_array(x_train[:,:,:,0])
    x_test = morphoMaths.AMD_in_one_array(x_test[:,:,:,0])
    d = datetime.date.today()
    strDate = d.strftime("%y_%m_%d")
    train_rec_errors = np.zeros(nb_run)
    test_rec_errors = np.zeros(nb_run)
    if svm:
        SVM_classification_accuracy = np.zeros(nb_run)
        shAE.train(x_train, nb_epochs=nb_epochs, X_val=x_test, verbose=2)
        shAE.save(path_to_model_directory="../ShallowAE/WithAMD/")
        train_rec_errors[idx] = np.mean(shAE.reconstruction_error(x_train))
        test_rec_errors[idx] = np.mean(shAE.reconstruction_error(x_test))
        np.save(out_path +'_training_errors_' + strDims, train_rec_errors)
        np.save(out_path +'_test_errors_' + strDims, test_rec_errors)
        if svm:
            SVM_classification_accuracy[idx] = shAE.best_SVM_classification_score(x_test, y_test)
            np.save(out_path +'_svm_acc_' + strDims, SVM_classification_accuracy)

import numpy as np
import bastien_utils
from shallowAE import ShallowAE
from sparseShallowAE import SparseShallowAE_KL, SparseShallowAE_L1, SparseShallowAE_KL_sum
import datetime
import pandas as pd
import morphoMaths

PATH_TO_MODELS_DIR = "../ShallowAE/WithAMD"
PATH_TO_DATA = "../"

def testShallowAEwithAMD(latent_dimensions=[100], nb_epochs=200, svm=False, path_to_dir = "../ShallowAE/WithAMD/"):
    x_train, _, x_test, y_test = bastien_utils.load_data(PATH_TO_DATA, train=True, test=True, subsetTest=False)
    x_train = morphoMaths.AMD_in_one_array(x_train[:,:,:,0])
    x_test = morphoMaths.AMD_in_one_array(x_test[:,:,:,0])
    d = datetime.date.today()
    strDims = str(latent_dimensions[0]) + "_" + str(latent_dimensions[-1]) 
    strDate = d.strftime("%y_%m_%d")
    out_path = path_to_dir + "/Simple/TestOutputs/" + strDate
    nb_run = len(latent_dimensions)
    train_rec_errors = np.zeros(nb_run)
    test_rec_errors = np.zeros(nb_run)
    train_kl_loss = np.zeros(nb_run)
    test_kl_loss = np.zeros(nb_run)
    sparsity_train = np.zeros(nb_run)
    sparsity_test = np.zeros(nb_run)
    max_approx_error_toOriginal_train = np.zeros(nb_run)
    max_approx_error_toOriginal_test = np.zeros(nb_run)
    max_approx_error_toRec_train = np.zeros(nb_run)
    max_approx_error_toRec_test = np.zeros(nb_run)
    np.save(out_path +'_dims_' + strDims, latent_dimensions)
    if svm:
        SVM_classification_accuracy = np.zeros(nb_run)
    for idx, latent_dim in enumerate(latent_dimensions):
        shAE = SparseShallowAE_KL_sum(latent_dim=latent_dimension, sparsity_weight=sp_w, sparsity_objective=sp_o)
        shAE.train(x_train, nb_epochs=nb_epochs, X_val=x_test, verbose=2)
        shAE.save()
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
        np.save(out_path +'_training_max_approx_error_toOriginal_dilatation' + strDims, max_approx_error_toOriginal_train)
        np.save(out_path +'_test_max_approx_error_toOriginal_dilation' + strDims, max_approx_error_toOriginal_test)
        np.save(out_path +'_training_max_approx_error_toRec_dilatation' + strDims, max_approx_error_toRec_train)
        np.save(out_path +'_test_max_approx_error_toRec_dilation' + strDims, max_approx_error_toRec_test)
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

