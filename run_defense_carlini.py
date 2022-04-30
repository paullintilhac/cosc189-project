import sys
import argparse
import os
import time

import numpy as np


import lasagne
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from lib.utils.data_utils import *
from lib.utils.model_utils import *
from lib.attacks.nn_attacks import *
from lib.defenses.nn_defenses import *


#-----------------------------------------------------------------------------#


def main(argv):
    """
    Main function of run_defense.py. Create adv. examples and evaluate attack.
    Implement defense and reevaluate the same attack (does not aware of the
    defense).
    """

    # Parameters

    batchsize = 500                         # Fixing batchsize
    no_of_mags = 1       # No. of deviations to consider
    dev_list = np.linspace(2, 2, no_of_mags)
    
    # Create model_dict from arguments
    model_dict = model_dict_create()
    print("model dict in run_defense: " + str(model_dict))
    # print("model dict: " + str(model_dict))
    # Load and parse specified dataset into numpy arrays
    print('Loading data...')
    dataset = model_dict['dataset']
    keras = model_dict['keras']
    print("USING KERAS MODEL FROM CARLINI? " + str(keras))
    if (dataset == 'MNIST'):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
            model_dict)
        # rd_list = [784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        rd_list = [model_dict["num_dims"]]
    elif dataset == 'GTSRB':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
            model_dict)
        rd_list = [1024, 338, 200, 100, 90, 80, 70, 60, 50, 40, 33, 30, 20, 10]
    elif dataset == 'HAR':
        X_train, y_train, X_test, y_test = load_dataset(model_dict)
        rd_list = [561, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        X_val = None
        y_val = None
    
    print("X_test shape: " + str(X_test.shape))
    print("X_test[0] == X_test[1]? " + str(np.array_equal(X_test[0],X_test[1])))
    print("running model setup")

    data_dict, test_prediction, dr_alg, X_test, sorted_distortions, attacked_predictions = \
        model_setup_keras(model_dict, X_train, y_train, X_test, y_test, X_val, y_val)


    import csv
    with open('sorted_distortions.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(sorted_distortions)):
            spamwriter.writerow([i]  + [sorted_distortions[i]])

    print("correct output shape: " + str(y_test.shape))

    layer_flag = None
    
    for rd in rd_list:
        print ("Starting strategic attack...")
        
        

#-----------------------------------------------------------------------------#

    
if __name__ == '__main__':
    main(sys.argv[1:])
#-----------------------------------------------------------------------------#
