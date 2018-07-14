import numpy as np
import pandas as pd
import bastien_utils
from shallowAE import ShallowAE
from sparseShallowAE import SparseShallowAE_KL, SparseShallowAE_L1, SparseShallowAE_KL_sum
import datetime

PATH_TO_MODELS_DIR = "../ShallowAE/"
PATH_TO_DATA = "../"

def test_KL_div(sparsity_weights = [1], sparsity_objectives = [0.1], latent_dimension=100, nb_epochs=200, svm=False, path_to_dir = "../ShallowAE/"):
    data = bastien_utils.load_data(PATH_TO_DATA, train=True, test=True, subsetTest=False)
    x_train, ytrain, x_test, y_test = data
    d = datetime.date.today()
    strDims = 'test_sparsity_hyperparameters' 
    strDate = d.strftime("%y_%m_%d")
    out_path = path_to_dir + "/Sparse/KL_div_sum/TestOutputs/" + strDate
    nb_sparsity_weights = len(sparsity_weights)
    nb_sparsity_objectives = len(sparsity_objectives)
    train_rec_errors = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    test_rec_errors = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    np.save(out_path +'_sparsityWeights_' + strDims, spar)
    np.save(out_path +'_sparsityObjectives_' + strDims, spar)
    if svm:
        SVM_classification_accuracy = np.zeros((nb_sparsity_weights, nb_sparsity_objectives))
    for idx1, sp_w in enumerate(sparsity_weights):
        for idx2, sp_o in enumerate(sparsity_objectives):
            shAE = SparseShallowAE_KL_sum(latent_dim=latent_dimension, sparsity_weight=sp_w, sparsity_objective=sp_o)
            shAE.train(x_train, nb_epochs=nb_epochs, X_val=x_test, verbose=2)
            shAE.save()
            train_rec_errors[idx1, idx2] = shAE.reconstruction_error(x_train)
            test_rec_errors[idx1, idx2] = shAE.reconstruction_error(x_test)
            np.save(out_path +'_training_errors_' + strDims, train_rec_errors)
            np.save(out_path +'_test_errors_' + strDims, test_rec_errors)
            if svm:
                SVM_classification_accuracy[idx1, idx2] = shAE.best_SVM_classification_score(x_test, y_test, nb_values_C=10, nb_values_gamma=10)
                np.save(out_path +'_svm_acc_' + strDims, SVM_classification_accuracy)
