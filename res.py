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
from keras.backend.tensorflow_backend import set_session

# Limit GPU usage
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


# Defined a metric for determining model accuracy
class Metrics(Callback):
    def __init__(self):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
            
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average = None)
        _val_recall = recall_score(val_targ, val_predict, average = None)
        _val_precision = precision_score(val_targ, val_predict, average = None)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        #print(len(_val_precision))
        print ("-- val_f1: " + str(_val_f1) +" — val_precision: "  + str(_val_precision) + "  — val_recall: " + str(_val_recall))
        return
                       
metrcs = Metrics()



DATA = 2200000

print('loading training data')
trainmat = h5py.File('../data/res_train.mat')
xtr = np.transpose(np.array(trainmat['trainxdata'][:,:,DATA:]), axes=(2,0,1))
ytr = np.array(trainmat['traindata'][0,DATA:]).T
print('finished loading training data')

print('loading validation data')
validmat = scipy.io.loadmat('../data/res_valid.mat')
print('finished loading validation data')

print(np.sum(validmat['validdata'][:,0]))
x = Input(shape = (1000,4))
conv1 = Convolution1D(filters=2,
                        padding="same",
                        activation="relu",
                        kernel_size=26,
                         kernel_initializer=K.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None))
conv2 = Convolution1D(filters=2,
                        padding="same",
                        activation="relu",
                        kernel_size=26,
                         kernel_initializer=K.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None))
conv3 = Convolution1D(filters=2,
                        padding="same",
                        activation="relu",
                        kernel_size=26, 
                     kernel_initializer=K.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None))
max1 = MaxPooling1D(pool_size=13, strides=13)
dropout1 = Dropout(0.2)
conv4 = Convolution1D(filters=2,
                        padding="same",
                        activation="relu",
                        kernel_size=13,
                         kernel_initializer=K.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None))
conv5 = Convolution1D(filters=2,
                        padding="same",
                        activation="relu",
                        kernel_size=13,
                         kernel_initializer=K.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None))
max2 = MaxPooling1D(pool_size=5, strides = 2)
dropout2 = Dropout(0.2)
flatten1 = Flatten()
dense1 = Dense(32)
dense2 = Dense(10, activation = 'relu')
dense3 = Dense(1)
sigmoid = Activation('sigmoid')

# Create the model connections
y1 = conv1(x)
y2 = conv2(y1)
y22 = conv3(y2)
y3 = K.layers.add([y1, y22])
y4 = max1(y3)
y5 = dropout1(y4)
y6 = conv4(y5)
y7 = conv5(y6)
y8 = K.layers.add([y5, y7])
y9 = max2(y8)
y10 = dropout2(y9)
y11 = flatten1(y10)
y12 = dense1(y11)
y13 = dense2(y12)
y14 = dense3(y13)
output = sigmoid(y14)


model = Model(x, output)



model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="multiple_adam.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
print("model training")

#model.fit(xtr,ytr, batch_size = 512, epochs = 60, shuffle = True)
model.fit(xtr, ytr, batch_size=512, epochs=60, shuffle=True, validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)), validmat['validdata'][:,0]), callbacks=[checkpointer,earlystopper], )



#print(model.summary())
#vale = np.transpose(validmat['validxdata'],axes=(0,2,1))
#a = model.predict(vale)
#print(a)
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')
"""
a = model.predict(xtr[:4], ytr[:4])
for i in range(40):
    for j in range(919):
        print(a[i][j],ytr[i][j])
"""


