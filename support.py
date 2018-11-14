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

print('loading training data')
trainmat = h5py.File('../data/train.mat')
xtr = np.transpose(np.array(trainmat['trainxdata'][:,:,:2000000]), axes=(2,0,1))
ytr = np.array(trainmat['traindata'][:,:2000000]).T
print('finished loading training data')
print('loading validation data')
validmat = scipy.io.loadmat('../data/valid.mat')

model = Sequential()
model.add(Convolution1D(input_dim=4,
                        input_length=1000,
                        nb_filter=320,
                        filter_length=26,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))

model.add(MaxPooling1D(pool_length=13, stride=13))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(input_dim=75*640, output_dim=925))
model.add(Activation('relu'))

model.add(Dense(input_dim=925, output_dim=919))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary", metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.fit(xtr, ytr, batch_size=100, nb_epoch=60, shuffle=True, validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)), validmat['validdata']), callbacks=[checkpointer,earlystopper])
