import tensorflow.compat.v1 as tf
import numpy as np
import time

from setup_mnist import MNIST, MNISTModel
from lib.utils.data_utils import *

from l2_attack import CarliniL2



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


if __name__ == "__main__":
    with tf.Session() as sess:
        # Create model_dict from arguments
        model_dict = model_dict_create()
        print("model dict in run_defense: " + str(model_dict))
        # print("model dict: " + str(model_dict))
        # Load and parse specified dataset into numpy arrays
        print('Loading data...')
        dataset = model_dict['dataset']
        keras = model_dict['keras']
        
        model =  MNISTModel("models/mnist", sess)

        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
        data_dict = get_data_shape(X_train, X_test, X_val)
        no_of_dim = data_dict['no_of_dim']


        X_test = np.transpose(X_test,axes = [0,2,3,1])

        y_onehot = np.zeros((len(y_test), 10))
        y_onehot[np.arange(len(y_test)), y_test] = 1
        
        data = (X_test, y_onehot)

        #data, model =  CIFAR(), CIFARModel("models/cifar", sess)
        attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
        #attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
        #                   largest_const=15)

        inputs, targets = generate_data(data, samples=1, targeted=True,
                                        start=0, inception=False)
        print("shape of inputs: " + str(inputs.shape) + ", shape of targets: " + str(targets.shape))
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        for i in range(len(adv)):
            print("Valid:")
            show(inputs[i])
            print("Adversarial:")
            show(adv[i])
            
            print("Classification:", model.model.predict(adv[i:i+1]))

            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)