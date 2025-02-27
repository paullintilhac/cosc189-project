import sys
import argparse
import os
import time

import numpy as np
import theano
import theano.tensor as T

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
    no_of_mags = 2                     # No. of deviations to consider
    dev_list = np.linspace(2, 4, no_of_mags)

    # Create model_dict from arguments
    model_dict = model_dict_create()
    print("model dict in run_defense: " + str(model_dict))
    # print("model dict: " + str(model_dict))
    # Load and parse specified dataset into numpy arrays
    print('Loading data...')
    dataset = model_dict['dataset']
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

    # Center data by subtracting mean of training set
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_test -= mean
    if (dataset == 'MNIST') or (dataset == 'GTSRB'):
        X_val -= mean

    print("running model setup")
    data_dict, test_prediction, dr_alg, X_test, input_var, target_var = \
        model_setup(model_dict, X_train, y_train, X_test, y_test, X_val, y_val)
    
    
    # print_output(model_dict, output_list, dev_list)
    # save_images(model_dict, data_dict, X_test, adv_x_ini, dev_list)

    layer_flag = None
    

    
    print("using defense "+str(model_dict['defense']))
    # Run defense
    defense = model_dict['defense']
    if defense != None:
        for rd in rd_list:
            print ("Starting strategic attack...")
            
            adv_x_ini, output_list = attack_wrapper(model_dict, data_dict, input_var,
                    target_var, test_prediction, dev_list, X_test, y_test, mean,
                                            dr_alg, rd)
            if defense == 'recons':
                recons_defense(model_dict, data_dict, input_var, target_var,
                               test_prediction, dev_list, adv_x_ini, rd,
                               X_train, y_train, X_test, y_test)
            elif defense == 'retrain':
                retrain_defense(model_dict, dev_list, adv_x_ini, rd, X_train,
                                y_train, X_test, y_test, X_val, y_val)
#-----------------------------------------------------------------------------#


if __name__ == '__main__':
    main(sys.argv[1:])
#-----------------------------------------------------------------------------#
