#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:54:49 2023

@author: george
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import math
import numpy as np
import numpy.linalg as nla
import pandas as pd
import re
import six
from os.path import join
from matplotlib import pyplot as plt


# #%% Training set
# #Example flights (SR-20)
# xtrain_sr = []

# sr20_folder = './../XC/JAISdata/SR-20' 
# for file in os.listdir(sr20_folder):
#     if file.startswith('log'):
#         print(file)
#         newflight = pd.read_csv(sr20_folder+'/'+file, header = 2 )
#         newflight = newflight.apply(pd.to_numeric, errors='coerce').fillna(0)
#         newflight = newflight[['  AltMSL','    IAS','    VSpd','  Pitch',' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4',
#                     ' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4',' E1 RPM']]
#         newflight['avg_CHT'] = newflight[[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4']].mean(axis=1) #############!!!!!
#         newflight=newflight.drop(columns=[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4'])

#         newflight['avg_EGT'] = newflight[[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4']].mean(axis=1) #############!!!!!
#         newflight=newflight.drop(columns=[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4'])

#         newflight.columns = newflight.columns.str.replace(' ', '')
#         newflight = newflight[newflight.IAS > 0]
#         newflight = newflight.reset_index(drop=True)
#         newflight['dAlt'] = newflight['AltMSL'].diff()
#         newflight['dAlt'][0] = 0
#         newflight['MA_VSpd'] = newflight['VSpd'].rolling(20).mean()
#         newflight['MA_VSpd'][0:20] = newflight['VSpd'][0:20]
#         if np.size(xtrain_sr,0) == 0:
#             xtrain_sr = newflight
#         else:
#             xtrain_sr = pd.concat([xtrain_sr, newflight])
        
# #Example flights (C-172)  
# xtrain_c = []

# c172_folder = './../XC/JAISdata/C-172' 
# for file in os.listdir(c172_folder):
#     if file.startswith('log'):
#         print(file)
#         newflight = pd.read_csv(c172_folder+'/'+file, header = 2 )
#         newflight = newflight.apply(pd.to_numeric, errors='coerce').fillna(0)
#         newflight = newflight[['  AltMSL','    IAS','    VSpd','  Pitch',' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4',
#                     ' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4',' E1 RPM']]
#         newflight['avg_CHT'] = newflight[[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4']].mean(axis=1) #############!!!!!
#         newflight=newflight.drop(columns=[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4'])

#         newflight['avg_EGT'] = newflight[[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4']].mean(axis=1) #############!!!!!
#         newflight=newflight.drop(columns=[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4'])

#         newflight.columns = newflight.columns.str.replace(' ', '')
#         newflight = newflight[newflight.IAS > 0]
#         newflight = newflight.reset_index(drop=True)
#         newflight['dAlt'] = newflight['AltMSL'].diff()
#         newflight['dAlt'][0] = 0
#         newflight['MA_VSpd'] = newflight['VSpd'].rolling(20).mean()
#         newflight['MA_VSpd'][0:20] = newflight['VSpd'][0:20]        
#         if np.size(xtrain_c,0) == 0:
#             xtrain_c = newflight
#         else:
#             xtrain_c = pd.concat([xtrain_c, newflight])

# x_train = pd.concat([xtrain_c, xtrain_sr])

# #Export
# np.save('./xtrain.npy',x_train)

# #%% Test/validation flights
# x_test = []

# test_folder = './../XC/JAISdata/testflights' 
# for file in os.listdir(test_folder):
#     if file.startswith('log'):
#         print(file)
#         newflight = pd.read_csv(test_folder+'/'+file, header = 2 )
#         newflight = newflight.apply(pd.to_numeric, errors='coerce').fillna(0)
#         newflight = newflight[['  AltMSL','    IAS','    VSpd','  Pitch',' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4',
#                     ' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4',' E1 RPM']]
#         newflight['avg_CHT'] = newflight[[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4']].mean(axis=1) #############!!!!!
#         newflight=newflight.drop(columns=[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4'])

#         newflight['avg_EGT'] = newflight[[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4']].mean(axis=1) #############!!!!!
#         newflight=newflight.drop(columns=[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4'])

#         newflight.columns = newflight.columns.str.replace(' ', '')
#         newflight = newflight[newflight.IAS > 0]
#         newflight = newflight.reset_index(drop=True)
#         newflight['dAlt'] = newflight['AltMSL'].diff()
#         newflight['dAlt'][0] = 0
#         newflight['MA_VSpd'] = newflight['VSpd'].rolling(20).mean()
#         newflight['MA_VSpd'][0:20] = newflight['VSpd'][0:20]          
#         if np.size(x_test,0) == 0:
#             x_test = newflight
#         else:
#             x_test = pd.concat([x_test, newflight])
            
        
# # #Export
# np.save('./xtest.npy',x_test)


from sklearn.preprocessing import StandardScaler



#%%HPC starts from here

x_train = np.load('./xtrain.npy')
x_test = np.load('./xtest.npy')


scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.fit_transform(x_test)

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout


## Building the autoencoder
n_inputs=np.shape(x_train)[1] 

inputs = Input(shape=(n_inputs,))
#encoder block 0
e0 = Dense(9)(inputs)
e0 = BatchNormalization()(e0)
e0 = Activation('selu')(e0)
#encoder block 1
e1 = Dense(8)(e0)
e1 = BatchNormalization()(e1)
e1 = Activation('selu')(e1)
#encoder block 2
e2 = Dense(7)(e1)
e2 = BatchNormalization()(e2)
e2 = Activation('selu')(e2)
#encoder block 3
e3 = Dense(6)(e2)
e3 = BatchNormalization()(e3)
e3 = Activation('selu')(e3)
#encoder block 4
e4 = Dense(5)(e3)
e4 = BatchNormalization()(e4)
e4 = Activation('selu')(e4)

#Bottleneck
h1 = Dense(4)(e4)
h1 = BatchNormalization()(h1)
h1 = Activation('selu')(h1)

#decoder block 0
d0 = Dense(5)(h1)
d0 = BatchNormalization()(d0)
d0 = Activation('selu')(d0)
#decoder block 1
d1 = Dense(6)(d0)
d1 = BatchNormalization()(d1)
d1 = Activation('selu')(d1)
#decoder block 2
d2 = Dense(7)(d1)
d2 = BatchNormalization()(d2)
d2 = Activation('selu')(d2)
#decoder block 3
d3 = Dense(8)(d2)
d3 = BatchNormalization()(d3)
d3 = Activation('selu')(d3)
#decoder block 4
d4 = Dense(8)(d3)
d4 = BatchNormalization()(d4)
d4 = Activation('selu')(d4)

outputs = Dense(n_inputs, activation = 'selu')(d4)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer=Adam(1e-5), loss='mse')
autoencoder.summary()

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping


#save the model when val_loss improves during training
checkpoint = ModelCheckpoint('model_params_876545678.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
csvlog = CSVLogger('train_log_876545678.csv',append=True)
#stop training if no improvement has been seen on val_loss for a while
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30)

## Training the autoencoder
training = autoencoder.fit(
    x_train_s,
    x_train_s,
    batch_size = 32,
    epochs=500,
    validation_data=(x_test_s, x_test_s),
    verbose=2,
    initial_epoch=0,
    callbacks=[checkpoint, csvlog, early_stopping]
    )


#encoder only
encoderonly = Model(inputs=inputs, outputs=h1)
# save the encoder to file
encoderonly.save('encoder_876545678.hdf5')



