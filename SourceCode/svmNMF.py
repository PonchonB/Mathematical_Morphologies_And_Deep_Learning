import numpy as np
import pandas as pd
from scipy import io
import metrics
import bastien_utils

FILE_PATH = "../"
x_test, y_test = bastien_utils.load_data(FILE_PATH, train=False, test=True, subsetTest=False)
print('x_test shape:', x_test.shape)
data = io.loadmat('../NMF/18_08_22_sparseNMF_dim100_spH0.6')

W = data['W']
H = data['H']
del data
print('W: ', W.shape)
print('H: ', H.shape)

atoms = W.transpose().reshape(100,28,28,1)
H = H.transpose()
del W

svm_score, best_params = metrics.best_SVM_classification_score(H, y_test, nb_values_C=10, nb_values_gamma=10)

SVM_best_C_parameter = best_params['C']
SVM_best_gamma_parameter = best_params['gamma']

results = pd.DataFrame(data={'SVM_classification_score':[svm_score],'SVM_best_C':[SVM_best_C_parameter], 'SVM_best_gamma':[SVM_best_gamma_parameter]})
results.to_csv('../NMF/18_08_27_SVMresults')


