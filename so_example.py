# Create Keras model
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D
import tensorflow.compat.v1 as tf
#from tensorflow.python.keras.backend import set_session
from tensorflow.compat.v1.keras.backend import set_session
import time
import numpy as np
tf.disable_v2_behavior()
tf.disable_eager_execution()

model_path ="/Users/paullintilhac/defense_research/cosc189-project/nn_models/MNIST/cnn_9_papernotsigmoid"

with np.load(model_path + '.npz') as f:
    param_values = [np.float32(f['arr_%d' % i])
                    for i in range(len(f.files))]

model = Sequential()
model.add(Conv2D(32, (5, 5),padding = "same",
                    input_shape=(28,28, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (5, 5),padding = "same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5),padding = "same"))
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5),padding = "same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(10))

new_weights=[]

#transpose weights from theano format to keras format
for i in range(len(param_values)):
    if (len(param_values[i].shape)==4):
        temp_weight =np.transpose(param_values[i],axes = [2,3,1,0])
    else: temp_weight = param_values[i]
    new_weights.append(temp_weight)

model.set_weights(new_weights)      

print("executing eagerly? " + str(tf.executing_eagerly()))

with tf.Session() as sess:
    set_session(sess)
    
    shape = (9,28,28,1)
    
    # the variable we're going to optimize over
    modifier = tf.Variable(np.zeros(shape,dtype=np.float32))

    # these are variables to be more efficient in sending data to tf
    timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
    
    # the resulting image, tanh'd to keep bounded from boxmin to boxmax
    boxmin = -0.5
    boxmax = 0.5
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.
    newimg = tf.tanh(modifier + timg) * boxmul + boxplus
    init = tf.global_variables_initializer()
    sess.run(init)
    # prediction BEFORE-SOFTMAX of the model
    output = model.predict(newimg,steps=9)
    print(output)
    

