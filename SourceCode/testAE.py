import numpy as np
import bastien_utils
from shallowAE import ShallowAE
from sparseShallowAE import SparseShallowAE_KL_sum
from nonNegSparseShallowAE import Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint, Sparse_NonNeg_ShallowAE_KLsum_AsymDecay
from nonNegShallowAE import NonNegShallowAE_NonNegConstraint
import datetime
import pandas as pd
import morphoMaths
from MaxPlusDecoder.ShallowAE_maxPlusDecoder.shallowAE_maxplus_NonNeg import NonNeg_ShallowAE_MaxPlus_Between0and1Constraint

PATH_TO_DATA = "../"

def testDims(ShallowAE_class=ShallowAE, latent_dimensions=[100], nb_epochs=400, nb_input_channels=1, one_channel_output=True,
            AMD=False, PADO=False, AMD_step=1, AMD_init_step=1, add_original_images=True,
            svm=False, path_to_dir = "../Results/ShallowAE/", **kwargs):
    original_images_train, _, original_images_test, y_test = bastien_utils.load_data_fashionMNIST(PATH_TO_DATA, train=True, test=True, subsetTest=False)
    if (nb_input_channels>1):
        if AMD:
            if (add_original_images & (nb_input_channels>2)):
                x_train = morphoMaths.AMD_in_one_array(original_images_train[:,:,:,0], levels=nb_input_channels-2, step=AMD_step, init_step=AMD_init_step, add_original_images=add_original_images)
                x_test = morphoMaths.AMD_in_one_array(original_images_test[:,:,:,0], levels=nb_input_channels-2, step=AMD_step, init_step=AMD_init_step, add_original_images=add_original_images)
                path_to_dir=path_to_dir+"/SeveralChannels/WithAMD/"
            else:
                x_train = morphoMaths.AMD_in_one_array(original_images_train[:,:,:,0], levels=nb_input_channels-1, step=AMD_step, init_step=AMD_init_step, add_original_images=add_original_images)
                x_test = morphoMaths.AMD_in_one_array(original_images_test[:,:,:,0], levels=nb_input_channels-1, step=AMD_step, init_step=AMD_init_step, add_original_images=add_original_images)
                path_to_dir=path_to_dir+"/SeveralChannels/WithAMD_NoOriginals/"
            x_train = bastien_utils.rescale_all_channels_between_0_and_1(x_train)
            x_test = bastien_utils.rescale_all_channels_between_0_and_1(x_test)
        else:
            if PADO:
                if (add_original_images & (nb_input_channels>2)):
                    x_train = morphoMaths.positive_decomposition_by_openings_by_rec(original_images_train[:,:,:,0], levels=nb_input_channels-2, step=AMD_step, init_step=AMD_init_step, add_original_images=add_original_images)
                    x_test = morphoMaths.positive_decomposition_by_openings_by_rec(original_images_test[:,:,:,0], levels=nb_input_channels-2, step=AMD_step, init_step=AMD_init_step, add_original_images=add_original_images)
                    path_to_dir=path_to_dir+"/SeveralChannels/WithPADO/"
                else:
                    x_train = morphoMaths.positive_decomposition_by_openings_by_rec(original_images_train[:,:,:,0], levels=nb_input_channels-1, step=AMD_step, init_step=AMD_init_step, add_original_images=add_original_images)
                    x_test = morphoMaths.positive_decomposition_by_openings_by_rec(original_images_test[:,:,:,0], levels=nb_input_channels-1, step=AMD_step, init_step=AMD_init_step, add_original_images=add_original_images)
                    path_to_dir=path_to_dir+"/SeveralChannels/WithPADO_NoOriginals/"
                x_train = bastien_utils.rescale_all_channels_between_0_and_1(x_train)
                x_test = bastien_utils.rescale_all_channels_between_0_and_1(x_test)    
            else:
                x_train = np.tile(x_train, (1,1,1,nb_input_channels))
                x_test = np.tile(x_test, (1,1,1,nb_input_channels))
                path_to_dir=path_to_dir+"/SeveralChannels/NoAMD/"
    else:
        x_train=original_images_train
        x_test=original_images_test
    if not one_channel_output:
        original_images_train=x_train
        original_images_test=x_test
    d = datetime.date.today()
    strDims = str(latent_dimensions[0]) + "_" + str(latent_dimensions[-1]) 
    strDate = d.strftime("%y_%m_%d")
    if ShallowAE_class==SparseShallowAE_KL_sum:
        out_path = path_to_dir + "/Sparse/KL_div_sum/TestOutputs/" + strDate
    elif ShallowAE_class==NonNegShallowAE_NonNegConstraint:
         out_path = path_to_dir + "/NonNegativity/NonNegConstraint/TestOutputs/" + strDate
    elif ShallowAE_class==Sparse_NonNeg_ShallowAE_KLsum_NonNegConstraint:
        out_path = path_to_dir + "/Sparse_NonNeg/KLdivSum_NonNegConstraint/TestOutputs/" + strDate
    elif ShallowAE_class==Sparse_NonNeg_ShallowAE_KLsum_AsymDecay:
        out_path = path_to_dir + "/Sparse_NonNeg/KLdivSum_AsymDecay/TestOutputs/" + strDate
    elif ShallowAE_class==NonNeg_ShallowAE_MaxPlus_Between0and1Constraint:
        out_path = path_to_dir + "/NonNegativity/Between0and1Constraint/TestOutputs/" + strDate
    else:
        out_path = path_to_dir + "/Simple/TestOutputs/" + strDate
    nb_run = len(latent_dimensions)
    train_rec_errors = np.zeros(nb_run)
    test_rec_errors = np.zeros(nb_run)
    sparsity_train = np.zeros(nb_run)
    sparsity_test = np.zeros(nb_run)
    max_approx_error_toOriginal_train = np.zeros(nb_run)
    max_approx_error_toOriginal_test = np.zeros(nb_run)
    max_approx_error_toRec_train = np.zeros(nb_run)
    max_approx_error_toRec_test = np.zeros(nb_run)
    np.save(out_path +'_dims_' + strDims, latent_dimensions)
    if svm:
        SVM_classification_accuracy = np.zeros(nb_run)
        SVM_best_C_parameter = np.zeros(nb_run)
        SVM_best_gamma_parameter = np.zeros(nb_run)
    for idx, d in enumerate(latent_dimensions):
        shAE = ShallowAE_class(latent_dim=d, nb_input_channels=nb_input_channels, one_channel_output=one_channel_output, **kwargs)
        shAE.train(X_train=x_train, X_train_expected_output=original_images_train, nb_epochs=nb_epochs, 
                    X_val=(x_test, original_images_test), verbose=2)
        shAE.save(path_to_model_directory=path_to_dir)
        train_rec_errors[idx] =shAE.reconstruction_error(x_train, X_rec_th=original_images_train)
        test_rec_errors[idx] = shAE.reconstruction_error(x_test, X_rec_th=original_images_test)
        sparsity_train[idx] = shAE.sparsity_measure(x_train)
        sparsity_test[idx] = shAE.sparsity_measure(x_test)
        max_approx_train = shAE.max_approximation_error(x_train, morphoMaths.dilatation, original_images=original_images_train, apply_to_bias=False, SE_scale=1)
        max_approx_error_toOriginal_train[idx] = max_approx_train[0]
        max_approx_error_toRec_train[idx] = max_approx_train[1]
        max_approx_test = shAE.max_approximation_error(x_test, morphoMaths.dilatation, original_images=original_images_test, apply_to_bias=False, SE_scale=1)
        max_approx_error_toOriginal_test[idx] = max_approx_test[0]
        max_approx_error_toRec_test[idx] = max_approx_test[1]
        np.save(out_path +'_training_errors_' + strDims, train_rec_errors)
        np.save(out_path +'_test_errors_' + strDims, test_rec_errors)
        np.save(out_path +'_training_sparsity_' + strDims, sparsity_train)
        np.save(out_path +'_test_sparsity_' + strDims, sparsity_test)
        np.save(out_path +'_training_max_approx_error_toOriginal_dilatation_' + strDims, max_approx_error_toOriginal_train)
        np.save(out_path +'_test_max_approx_error_toOriginal_dilation_' + strDims, max_approx_error_toOriginal_test)
        np.save(out_path +'_training_max_approx_error_toRec_dilatation_' + strDims, max_approx_error_toRec_train)
        np.save(out_path +'_test_max_approx_error_toRec_dilation_' + strDims, max_approx_error_toRec_test)
        if svm:
            SVM_classif = shAE.best_SVM_classification_score(x_test, y_test, nb_values_C=10, nb_values_gamma=10)
            SVM_classification_accuracy[idx] = SVM_classif[0]
            SVM_best_C_parameter[idx] = SVM_classif[1]['C']
            SVM_best_gamma_parameter[idx] = SVM_classif[1]['gamma']
            np.save(out_path +'_svm_acc_' + strDims, SVM_classification_accuracy)
            np.save(out_path +'_svm_best_C_' + strDims, SVM_best_C_parameter)
            np.save(out_path +'_svm_best_gamma_' + strDims, SVM_best_gamma_parameter)
    if svm:
        results = pd.DataFrame(data={'dimension':latent_dimensions,
                                'training_error':train_rec_errors, 'test_error':test_rec_errors,
                                'training_sparsity':sparsity_train, 'test_sparsity':sparsity_test,
                                'training_max_approx_error_toOriginal_dilatation':max_approx_error_toOriginal_train,
                                'test_max_approx_error_toOriginal_dilatation':max_approx_error_toOriginal_test,
                                'training_max_approx_error_toRec_dilatation':max_approx_error_toRec_train,
                                'test_max_approx_error_toRec_dilatation':max_approx_error_toRec_test,
                                'SVM_classification_score':SVM_classification_accuracy,
                                'SVM_best_C':SVM_best_C_parameter, 'SVM_best_gamma':SVM_best_gamma_parameter})
    else:
            results = pd.DataFrame(data={'dimension':latent_dimensions,
                                'training_error':train_rec_errors, 'test_error':test_rec_errors,
                                'training_sparsity':sparsity_train, 'test_sparsity':sparsity_test,
                                'training_max_approx_error_toOriginal_dilatation':max_approx_error_toOriginal_train,
                                'test_max_approx_error_toOriginal_dilatation':max_approx_error_toOriginal_test,
                                'training_max_approx_error_toRec_dilatation':max_approx_error_toRec_train,
                                'test_max_approx_error_toRec_dilatation':max_approx_error_toRec_test})
    results.to_csv(out_path+'results')
