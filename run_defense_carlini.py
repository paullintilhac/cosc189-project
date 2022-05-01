import sys
import argparse
import os
from sklearn.decomposition import PCA
import tensorflow.compat.v1 as tf
import numpy as np
import time
from setup_mnist import MNIST, MNISTModel
from lib.utils.data_utils import *
from l2_attack import CarliniL2
import csv
from lib.attacks.nn_attacks import *
from lib.defenses.nn_defenses import *
from lib.utils.data_utils import *
from train_models import train

#-----------------------------------------------------------------------------#


def main(argv):
    """
    Main function of run_defense.py. Create adv. examples and evaluate attack.
    Implement defense and reevaluate the same attack (does not aware of the
    defense).
    """
    model_dict = model_dict_create()
    print("model_dict: " + str(model_dict))
    with tf.Session() as sess:
        model =  MNISTModel("models/mnist", sess)
        # Create model_dict from arguments
        print("model dict in run_defense: " + str(model_dict))
        # print("model dict: " + str(model_dict))
        # Load and parse specified dataset into numpy arrays
        print('Loading data...')
        dataset = model_dict['dataset']
        keras = model_dict['keras']
        rev = model_dict['rev']
        dim_red = model_dict['dim_red']
        small = model_dict['small']
        gamma = model_dict['gamma']
        kernel = model_dict['kernel']
        rd = model_dict["num_dims"]
        print("rd in model_setup: " + str(rd))
    

        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
        data_dict = get_data_shape(X_train, X_test, X_val)
        no_of_dim = data_dict['no_of_dim']


        X_test_transpose = np.transpose(X_test,axes = [0,2,3,1])-.5

        print(" first image shape: " + str(X_test[0].shape))
        print(" min: " + str(np.min(X_test_transpose[0])))
        print(" max: " + str(np.max(X_test_transpose[0])))
        y_onehot = np.zeros((len(y_test), 10))
        y_onehot[np.arange(len(y_test)), y_test] = 1
        
        # Defining symbolic variable for network output
        test_prediction = model.predict(X_test_transpose)

        # max_index_col = np.argmax(test_prediction, axis=0)
        # print("max_index_col shape: " + str(max_index_col.shape))

        test_prediction_array = tf.math.argmax(test_prediction,axis=1).eval()

        print("executing eagerly? " + str(tf.executing_eagerly()))

        print("prediction shape: " + str(test_prediction.shape))        
        print("y_test_shape: " + str(y_test.shape))
        print("first prediction: " + str(test_prediction[0].eval()))
        print("first label: " + str(y_test[0]))
        print("test_prediction_array shape:  " + str(test_prediction_array.shape))
        print("test_prediction_array[0]: " + str(test_prediction_array[0]))

        accuracy = np.mean(test_prediction_array == y_test)
        print("undefended, unattacked test accuracy: " + str(accuracy))

        data = (X_test_transpose, y_onehot)

        #data, model =  CIFAR(), CIFARModel("models/cifar", sess)
        attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0, targeted=False)
        #attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
        #                   largest_const=15)

        inputs, targets = generate_data(data, samples=9, targeted=False,
                                        start=0, inception=False)

        print("shape of inputs: " + str(inputs.shape) + ", shape of targets: " + str(targets.shape))
        #print("targets: " + str(targets))

        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        
        distortion = np.sum((adv - inputs)**2,axis=(1,2,3))**.5
        print("distortion shape: " + str(distortion.shape))
        
        sorted_distortion = np.sort(distortion)
        attacked_predictions = model.model.predict(adv)
        attacked_prediction_array = tf.math.argmax(attacked_predictions,axis=1).eval()

        attacked_accuracy = np.mean(attacked_prediction_array == y_test)
        max_distortion = np.max(distortion)
        mean_distortion = np.mean(distortion)
        print("undefended, attacked accuracy: " + str(attacked_accuracy))
        print("max distortion: " + str(max_distortion))
        print("mean distortion: " + str(mean_distortion))

        with open('sorted_distortions.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(sorted_distortion)):
                spamwriter.writerow([i]  + [sorted_distortion[i]])


        #run dimension reduction
        X_train_t, X_test_t, X_val_t, dr_alg = dr_wrapper(X_train, X_test, X_val, dim_red, rd, y_train, rev,small, gamma, kernel)
        

        class Object(object):
            pass
        
        X_test_t = np.transpose(X_test,axes = [0,2,3,1])-.5

        print("X_test_t.shape: " + str(X_test_t.shape))
        print(" first image shape: " + str(X_test_t[0].shape))
        print(" min: " + str(np.min(X_test_t[0])-.5))
        print(" max: " + str(np.max(X_test_t[0])-.5))
        data = Object()
        data.train_data = np.transpose(X_train_t,axes = [0,2,3,1])-.5
        data.train_labels = y_train
        data.validation_data = np.transpose(X_val_t,axes = [0,2,3,1])-.5
        data.validation_labels = y_val
        # re-train model on transformed data
        # this needs to be cleaned up /refactored so that it is using automatic naming conventions like before, 
        # and checking to see if model already exists, etc. 
        #train(data, "models/retrain",[32, 32, 64, 64, 200, 200] , num_epochs=5)

        # once we have trained model, we load it
        defended_model =  MNISTModel("models/retrain", sess)

        # run prediction on defended, un-attacked model
        test_prediction_defended_unattacked = defended_model.predict(X_test_t)

        # max_index_col = np.argmax(test_prediction, axis=0)
        # print("max_index_col shape: " + str(max_index_col.shape))

        test_prediction_defended_unattacked_array = tf.math.argmax(test_prediction_defended_unattacked,axis=1).eval()

        defended_unattacked_accuracy = np.mean(test_prediction_defended_unattacked_array == y_test)
        print("defended, unattacked test accuracy: " + str(accuracy))

        defended_data = (X_test_t, y_onehot)

        #data, model =  CIFAR(), CIFARModel("models/cifar", sess)
        white_box_attack = CarliniL2(sess, defended_model, batch_size=9, max_iterations=1000, confidence=0, targeted=False)

        defended_inputs, defended_targets = generate_data(defended_data, samples=9, targeted=False,
                                        start=0, inception=False)

        print("shape of defended inputs: " + str(defended_inputs.shape) + ", shape of defended targets: " + str(defended_targets.shape))
        print("targets: " + str(defended_targets))

        timestart = time.time()
        defended_adv = white_box_attack.attack(defended_inputs, defended_targets)
        timeend = time.time()
        
        defended_attacked_distortion = np.sum((defended_adv - defended_inputs)**2,axis=(1,2,3))**.5
        print("distortion shape: " + str(defended_attacked_distortion.shape))
        
        defended_attacked_sorted_distortion = np.sort(defended_attacked_distortion)
        defended_attacked_predictions = defended_model.model.predict(defended_adv)
        defended_attacked_prediction_array = tf.math.argmax(defended_attacked_predictions,axis=1).eval()

        defended_attacked_accuracy = np.mean(defended_attacked_prediction_array == y_test)
        defended_attacked_max_distortion = np.max(defended_attacked_distortion)
        defended_attacked_mean_distortion = np.mean(defended_attacked_distortion)
        print("defended, attacked accuracy: " + str(defended_attacked_accuracy))
        print("max distortion: " + str(defneded_attacked_max_distortion))
        print("mean distortion: " + str(defended_attacked_mean_distortion))

        with open('defended_sorted_distortions.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(defended_attacked_sorted_distortion)):
                spamwriter.writerow([i]  + [defended_attacked_sorted_distortion[i]])


    #-----------------------------------------------------------------------------#
    
def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1, 1001), 10)
            else:
                seq = range(data[1].shape[1])

            for j in seq:
                if (j == np.argmax(data[1][start + i])) and (inception == False):
                    continue
                inputs.append(data[0][start + i])
                targets.append(np.eye(data[1].shape[1])[j])
        else:
            inputs.append(data[0][start + i])
            targets.append(data[1][start + i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#


if __name__ == '__main__':
    main(sys.argv[1:])
#-----------------------------------------------------------------------------#
