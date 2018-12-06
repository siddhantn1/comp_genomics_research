import numpy as np
import h5py
import scipy.io
import keras as K
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.engine.input_layer import Input
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.models import Model
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import metrics
import keras_metrics
import tensorflow as tf
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.losses.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.backend.mean(weighted_b_ce)

    return weighted_binary_crossentropy

model = load_model("multiple_adam.hdf5")

print('loading validation data')
validmat = scipy.io.loadmat('../data/res_valid.mat')
print('finished loading validation data')


preds = model.predict(np.transpose(validmat['validxdata'],axes=(0,2,1)))
print(preds.shape)
print(np.sum(preds))
print(np.max(preds))
              