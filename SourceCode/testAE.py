import numpy as np
import bastien_utils
from shallowAE import ShallowAE
import datetime
import pandas as pd
import morphoMaths

PATH_TO_DATA = "../"

def testDims(ShallowAE_class=ShallowAE, latent_dimensions=[100], nb_epochs=400, nb_input_channels=1, one_channel_output=True,
            AMD=False, AMD_step=1, AMD_init_step=1,
            svm=False, path_to_dir = "../ShallowAE/", **kwargs):
    x_train, _, x_test, y_test = bastien_utils.load_data(PATH_TO_DATA, train=True, test=True, subsetTest=False)
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
    strDims = str(latent_dimensions[0]) + "_" + str(latent_dimensions[-1]) 
    strDate = d.strftime("%y_%m_%d")
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
        shAE.train(x_train, nb_epochs=nb_epochs, X_val=x_test, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)])
        shAE.save(path_to_model_directory=path_to_dir)
        train_rec_errors[idx] =shAE.reconstruction_error(x_train)
        test_rec_errors[idx] = shAE.reconstruction_error(x_test)
        sparsity_train[idx] = shAE.sparsity_measure(x_train)
        sparsity_test[idx] = shAE.sparsity_measure(x_test)
        max_approx_train = shAE.max_approximation_error(x_train, morphoMaths.dilatation, apply_to_bias=True, SE_scale=1)
        max_approx_error_toOriginal_train[idx] = max_approx_train[0]
        max_approx_error_toRec_train[idx] = max_approx_train[1]
        max_approx_test = shAE.max_approximation_error(x_test, morphoMaths.dilatation, apply_to_bias=True, SE_scale=1)
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
                                'training_kl_loss':train_kl_loss, 'test_kl_loss':test_kl_loss,
                                'training_sparsity':sparsity_train, 'test_sparsity':sparsity_test,
                                'training_max_approx_error_toOriginal_dilatation':max_approx_error_toOriginal_train,
                                'test_max_approx_error_toOriginal_dilatation':max_approx_error_toOriginal_test,
                                'training_max_approx_error_toRec_dilatation':max_approx_error_toRec_train,
                                'test_max_approx_error_toRec_dilatation':max_approx_error_toRec_test})
    results.to_csv(out_path+'results')

