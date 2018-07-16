import numpy as np
import bastien_utils
from shallowAE import ShallowAE
from sparseShallowAE import SparseShallowAE_KL, SparseShallowAE_L1
import datetime

PATH_TO_MODELS_DIR = "../ShallowAE/"
PATH_TO_DATA = "../"

def testShallowAE(latent_dimensions=[100], nb_epochs=200, svm=False, path_to_dir = "../ShallowAE/"):
    data = bastien_utils.load_data(PATH_TO_DATA, train=True, test=True, subsetTest=False)
    x_train, ytrain, x_test, y_test = data
    d = datetime.date.today()
    strDims = str(latent_dimensions[0]) + "_" + str(latent_dimensions[-1]) 
    strDate = d.strftime("%y_%m_%d")
    out_path = path_to_dir + "/Sparse/KL_div/TestOutputs/" + strDate
    nb_run = len(latent_dimensions)
    train_rec_errors = np.zeros(nb_run)
    test_rec_errors = np.zeros(nb_run)
    np.save(out_path +'_dims_' + strDims, latent_dimensions)
    if svm:
        SVM_classification_accuracy = np.zeros(nb_run)
    for idx, latent_dim in enumerate(latent_dimensions):
        shAE = SparseShallowAE_KL(latent_dim=latent_dim)
        shAE.train(x_train, nb_epochs=nb_epochs, X_val=x_test, verbose=2)
        shAE.save()
        train_rec_errors[idx] = np.mean(shAE.reconstruction_error(x_train))
        test_rec_errors[idx] = np.mean(shAE.reconstruction_error(x_test))
        np.save(out_path +'_training_errors_' + strDims, train_rec_errors)
        np.save(out_path +'_test_errors_' + strDims, test_rec_errors)
        if svm:
            SVM_classification_accuracy[idx] = shAE.best_SVM_classification_score(x_test, y_test)
            np.save(out_path +'_svm_acc_' + strDims, SVM_classification_accuracy)

