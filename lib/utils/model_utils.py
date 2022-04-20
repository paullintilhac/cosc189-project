"""
This file contains helper functions that assist in managing NN models including
creating Lasagne model, loading parameters to model, saving parameters, and
setting up model.
"""

import sys
import os
import argparse
import numpy as np

from lib.utils.data_utils import *
from lib.utils.lasagne_utils import *
from lib.utils.theano_utils import *
from lib.utils.dr_utils import *

#------------------------------------------------------------------------------#


def model_creator(model_dict, data_dict, input_var, target_var, rd=None,
                  layer_flag=None):
    """
    Create a Lasagne model/network as specified in <model_dict> and check
    whether the model already exists in model folder.
    """
    print("creating model...")
    print(str(model_dict))
    n_epoch = model_dict['num_epochs']
    dataset = model_dict['dataset']
    model_name = model_dict['model_name']
    rev_flag = model_dict['rev']
    DR = model_dict['dim_red']
    n_out = model_dict['n_out']
    no_of_dim = data_dict['no_of_dim']
    print("no_of_dim in model creator: " + str(no_of_dim))
    # Determine input size
    if no_of_dim == 2:
        no_of_features = data_dict['no_of_features']
        in_shape = (None, no_of_features)
    elif no_of_dim == 3:
        channels = data_dict['channels']
        features_per_c = data_dict['features_per_c']
        in_shape = (None, channels, features_per_c)
    elif no_of_dim == 4:
        channels = data_dict['channels']
        height = data_dict['height']
        width = data_dict['width']
        in_shape = (None, channels, height, width)
    print("in shape: " + str(in_shape))
    #------------------------------- CNN model --------------------------------#
    if model_name == 'cnn':
        if n_epoch is not None:
            num_epochs = n_epoch
        else:
            num_epochs = 50
        depth = 9
        width = 'papernot'
        rate = 0.01
        activation = model_dict['nonlin']
        model_dict.update({'num_epochs': num_epochs, 'rate': rate,
                           'depth': depth, 'width': width})
        if rd is not None and not rev_flag:
          network = build_cnn_rd(input_var, rd)
        else:
          network = build_cnn(in_shape, n_out, input_var)

    #------------------------------- MLP model --------------------------------#
    elif model_name == 'mlp':
        if n_epoch is not None:
            num_epochs = n_epoch
        else:
            num_epochs = 500
        depth = 2
        width = 100
        rate = 0.01
        activation = model_dict['nonlin']
        model_dict.update({'num_epochs': num_epochs, 'rate': rate,
                           'depth': depth, 'width': width})
        if layer_flag:
            network, layers = build_hidden_fc(in_shape, n_out, input_var,
                                              activation, width)
        else:
            network, _ = build_hidden_fc(in_shape, n_out, input_var, activation,
                                         width)
        # if rd:
        #     network, _ = build_hidden_fc_rd(in_shape, n_out, input_var,
        #                                     activation, width, rd)

    #------------------------------ Custom model ------------------------------#
    elif model_name == 'custom':
        if n_epoch is not None:
            num_epochs = n_epoch
        else:
            num_epochs = 500
        depth = 2
        width = 100
        drop_in = 0.2
        drop_hidden = 0.5
        rate = 0.01
        activation = model_dict['nonlin']
        model_dict.update({'num_epochs': num_epochs, 'rate': rate,
                           'depth': depth, 'width': width, 'drop_in': drop_in,
                           'drop_hidden': drop_hidden})
        network = build_custom_mlp(in_shape, n_out, input_var, activation,
                                   int(depth), int(width), float(drop_in),
                                   float(drop_hidden))

    abs_path_m = resolve_path_m(model_dict)
    model_path = abs_path_m + get_model_name(model_dict, rd)
    model_exist_flag = 0
    if os.path.exists(model_path + '.npz'):
        model_exist_flag = 1

    if layer_flag:
        return network, model_exist_flag, layers
    else:
        return network, model_exist_flag
#------------------------------------------------------------------------------#


def model_loader(model_dict, rd=None):
    """
    Load parameters of the saved Lasagne model
    """
    print("loading model...")
    #print(str(model_dict))
    mname = get_model_name(model_dict, rd)
    abs_path_m = resolve_path_m(model_dict)

    model_path = abs_path_m + mname

    with np.load(model_path + '.npz') as f:
        param_values = [np.float32(f['arr_%d' % i])
                        for i in range(len(f.files))]
    return param_values
#------------------------------------------------------------------------------#


def model_saver(network, model_dict, rd=None):
    """
    Save model parameters in model foler as .npz file compatible with Lasagne
    """
    print("saving model")
    #print(str(model_dict))
    mname = get_model_name(model_dict, rd)
    abs_path_m = resolve_path_m(model_dict)

    model_path = abs_path_m + mname

    np.savez(model_path + '.npz', *lasagne.layers.get_all_param_values(network))
#------------------------------------------------------------------------------#


def model_setup(model_dict, X_train, y_train, X_test, y_test, X_val, y_val,
                rd=None, layer=None):
    """
    Main function to set up network (create, load, test, save)
    """
    print("setting up model")
    #print(str(model_dict))
    rev = model_dict['rev']
    dim_red = model_dict['dim_red']
    small = model_dict['small']
    gamma = model_dict['gamma']
    kernel = model_dict['kernel']
    print("rd in model_setup: " + str(rd))
    if rd:
        # Doing dimensionality reduction on dataset
        print("Doing {} with rd={} over the training data".format(dim_red, rd))
        X_train, X_test, X_val, dr_alg = dr_wrapper(X_train, X_test, X_val, dim_red, rd, y_train, rev,small, gamma, kernel)
    else:
        dr_alg = None
    print("dr_alg in model_setup: " + str(dr_alg))
    # Getting data parameters after dimensionality reduction

    print("X_train shape: " + str(X_train.shape))
    print("x_test shape: " + str(X_test.shape))
    print("X_Val shape: " + str(X_val.shape))

    data_dict = get_data_shape(X_train, X_test, X_val)
    no_of_dim = data_dict['no_of_dim']

    # Prepare Theano variables for inputs and targets
    if no_of_dim == 2:
        input_var = T.matrix('inputs')
    elif no_of_dim == 3:
        input_var = T.tensor3('inputs')
    elif no_of_dim == 4:
        input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Check if model already exists
    if layer:
        network, model_exist_flag, layers = model_creator(model_dict, data_dict,
                                                          input_var, target_var,
                                                          rd, layer)
    else:
        network, model_exist_flag = model_creator(model_dict, data_dict,
                                                  input_var, target_var, rd,
                                                  layer)

    # Defining symbolic variable for network output
    prediction = lasagne.layers.get_output(network)
    # Defining symbolic variable for network parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    # Defining symbolic variable for network output with dropout disabled
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # Building or loading model depending on existence
    if model_exist_flag == 1:
        # Load the correct model:
        print("model exists")
        param_values = model_loader(model_dict, rd)
        lasagne.layers.set_all_param_values(network, param_values)
    elif model_exist_flag == 0:
        print("model does not exist")
        # Launch the training loop.
        print("Starting training...")
        if layer is not None:
            model_trainer(input_var, target_var, prediction, test_prediction,
                          params, model_dict, X_train, y_train,
                          X_val, y_val, network, layers)
        else:
            model_trainer(input_var, target_var, prediction, test_prediction,
                          params, model_dict, X_train, y_train,
                          X_val, y_val, network)
        model_saver(network, model_dict, rd)

    # Evaluating on retrained inputs
    test_model_eval(model_dict, input_var, target_var, test_prediction,
                    X_test, y_test, rd)

    return data_dict, test_prediction, dr_alg, X_test, input_var, target_var
#------------------------------------------------------------------------------#


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


def model_setup_carlini(rd, model_dict, X_train, y_train, X_test, y_test, X_val,
                        y_val, mean, ax=None, layer=None):
    """
    Main function to set up network (create, load, test, save)
    """

    rev = model_dict['rev']
    dim_red = model_dict['dim_red']
    if rd != None:
        # Doing dimensionality reduction on dataset
        print("Doing {} with rd={} over the training data".format(dim_red, rd))
        X_train, X_test, X_val, dr_alg = dr_wrapper(X_train, X_test, X_val, dim_red, rd, y_train, rev)
    else:
        dr_alg = None
    mean = np.mean(X_train, axis=0)
    
    print("mean: " + str(mean.shape))
    print("X_test.shape: " + str(X_test.shape))
    # Getting data parameters after dimensionality reduction
    data_dict = get_data_shape(X_train, X_test, X_val)
    no_of_dim = data_dict['no_of_dim']
    # Prepare Theano variables for inputs and targets
    if no_of_dim == 2:
        input_var = T.tensor('inputs')
    elif no_of_dim == 3:
        input_var = T.tensor3('inputs')
    elif no_of_dim == 4:
        input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Check if model already exists
    if layer is not None:
        network, model_exist_flag, layers = model_creator(model_dict, data_dict,
                                            input_var, target_var, rd, layer)
    else:
        network, model_exist_flag = model_creator(model_dict, data_dict,
                                    input_var, target_var, rd, layer)

    # Defining symbolic variable for network output
    prediction = lasagne.layers.get_output(network)
    # Defining symbolic variable for network parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    # Defining symbolic variable for network output with dropout disabled
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # Building or loading model depending on existence
    if model_exist_flag == 1:
        print("model exists")
        # Load the correct model:
        param_values = model_loader(model_dict, rd)

        print("param values shape: " + str(len(param_values)))
        print("param values[0] shape: " + str(len(param_values[0])))

        #lasagne.layers.set_all_param_values(network, param_values)

        # Create Keras model
        from tensorflow.compat.v1.keras.models import Sequential
        from tensorflow.compat.v1.keras.layers import Dense, Dropout, Activation, Flatten
        from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D
        import tensorflow.compat.v1 as tf
        import time
        from l2_attack import CarliniL2
        tf.compat.v1.disable_eager_execution()



        model = Sequential()

        # paul note: this is what makes sense to me for handling the rd is none and not none cases,
        # but it doesn't actually mirror the original code for bhagoji. 
        # that code has two lines for the rd is not none case it looked like this:
        #   model.add(Dense(rd, activation=None,
        #                    input_shape=(784,), use_bias=False))
        #    model.add(Dense(100, activation='sigmoid'))
        # why? isn't this adding two layers where the rd is none case only adds one?
        # and why was the input shape here still 784? It could be wrong, but also worth thinking 
        # about what they were doing. I don't know keras very well so I can't say for sure.

        model.add(Conv2D(32, (5, 5),padding = "same",
                         input_shape=(28,28, 1)))

        #after this we should have input volume of size 24x24
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5, 5),padding = "same"))
        #20x20
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #10x10
        model.add(Conv2D(64, (5, 5),padding = "same"))
        #6x6
        model.add(Activation('relu'))
        model.add(Conv2D(64, (5, 5),padding = "same"))

        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(10))
        print("saheps")
        print(type(param_values))
        new_weights=[]
      
        for i in range(len(param_values)):
            print("param value shape: " + str(param_values[i].shape))
            if (len(param_values[i].shape)==4):
                temp_weight =np.transpose(param_values[i],axes = [2,3,1,0])
            else: temp_weight = param_values[i]
            print("temp weight shape: " + str(temp_weight.shape))
            new_weights.append(temp_weight)


        model.set_weights(new_weights)      

        if rd is not None:
            A = gradient_transform(model_dict, dr_alg)
            param_values = [A.T] + param_values

        # model.set_weights(param_values)
        # m_path = './keras/' + get_model_name(model_dict, rd)
        # model.save(m_path)
        # model.load_weights(m_path)

        y_onehot = np.zeros((len(y_test), 10))
        y_onehot[np.arange(len(y_test)), y_test] = 1
        # X_test was mean-subtracted before, now we add the mean back
        ## paul note: why are we subtracting .5 here?
        ## probably need conditional statement here to handle if rd is not none
        X_test_mean = (X_test + mean - 0.5).reshape(-1, 784)
        data = (X_test_mean, y_onehot)
        # paul note: also probly need conditional here for rd not none
        mean_flat = mean.reshape(-1, 784)

        # l2-Carlini Attack
        
        print("executing eagerly? " + str(tf.executing_eagerly()))

        with tf.Session() as sess:
            attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
            #attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
            #                   largest_const=15)

            inputs, targets = generate_data(data, samples=1, targeted=True,
                                            start=0, inception=False)
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
                # Test overall accuracy of the model
                pred = model.predict(inputs + 0.5 - mean_flat)
                correct = 0
                for i in range(pred.shape[0]):
                    if np.argmax(pred[i]) == y_test[i]:
                        correct += 1
                print('Overall accuracy on test images: ',
                    correct / float(pred.shape[0]))

                pred = model.predict(adv)
                correct = 0
                for i in range(pred.shape[0]):
                    if np.argmax(pred[i]) == y_test[i]:
                        correct += 1
                print('Overall accuracy on adversarial images: ',
                    correct / float(pred.shape[0]))

                dists_sorted = sorted(dists)

                for i in range(len(dists)):
                    plotfile.write('{} {} \n'.format(i, dists_sorted[i]))

                # Plot histogram
                # import matplotlib.pyplot as plt
                # dists = np.array(dists)
                # ax.hist(dists, 50, normed=1, histtype='step', cumulative=True,label=str(rd))


    elif model_exist_flag == 0:
        print("model does not exist")
        # Launch the training loop.
        print("Starting training...")
        if layer is not None:
            model_trainer(input_var, target_var, prediction, test_prediction,
                          params, model_dict, X_train, y_train, X_val, y_val,
                          network, layers)
        else:
            model_trainer(input_var, target_var, prediction, test_prediction,
                          params, model_dict, X_train, y_train, X_val, y_val,
                          network)
        model_saver(network, model_dict, rd)

    # Evaluating on retrained inputs
    test_model_eval(model_dict, input_var, target_var, test_prediction, X_test, y_test, rd)
