import numpy as np
import h5py
import scipy.io
import keras as k
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


DATA = 500000

print('loading training data')
trainmat = h5py.File('../data/train.mat')
xtr = np.transpose(np.array(trainmat['trainxdata'][:,:,:DATA]), axes=(2,0,1))
ytr = np.array(trainmat['traindata'][:,:DATA]).T
"""
denom = ytr.shape[0]
num = np.sum(ytr, axis = 0)
print(num.shape)
print("THIS IS THE BASELINE")

print(num/denom)
"""
print('finished loading training data')
print('loading validation data')
validmat = scipy.io.loadmat('../data/valid.mat')

model = Sequential()

model.add(Convolution1D(input_shape = (1000,4),
                        filters=320,
                        padding="valid",
                        activation="relu",
                        kernel_size=26))

model.add(MaxPooling1D(pool_size=13, strides=13))

model.add(Dropout(0.2))

model.add(Convolution1D(filters=320, padding="same", activation = "relu", kernel_size = 26))
model.add(MaxPooling1D(pool_size=5, strides = 2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(1800))
model.add(Dense(925))
#model.add(Dense(input_dim=75*640, units=925))
model.add(Activation('relu'))

model.add(Dense(input_dim=925, units=919))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="multiple_adam.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
print("model training")

#model.fit(xtr,ytr, batch_size = 512, epochs = 60, shuffle = True)
model.fit(xtr, ytr, batch_size=100, epochs=60, shuffle=True, validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)), validmat['validdata']), callbacks=[checkpointer,earlystopper])



print(model.summary())






