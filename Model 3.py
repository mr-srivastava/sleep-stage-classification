# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:55:55 2019

@author: Aadarsh Srivastava
"""

import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv1D,Flatten,Dropout,Dense
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import scipy.io

def read_data(file):
    eeg_fpz  = scipy.io.loadmat('D:/Project/Github_dwn/'+file+'/matlab/eeg_fpz.mat')
    eeg_pz  = scipy.io.loadmat('D:/Project/Github_dwn/'+file+'/matlab/eeg_fpz.mat')
    hyp = scipy.io.loadmat('D:/Project/Github_dwn/'+file+'/matlab/eeg_fpz.mat')
    return eeg_fpz,eeg_pz,hyp

def data_reshape(eeg_fpz,eeg_pz,hyp):
    fpz  = eeg_fpz['signal']
    fpz = np.reshape(fpz,(2650,3000,1))
    pz  = eeg_pz['signal']
    pz = np.reshape(pz,(2650,3000,1))
    label = hyp['hypnogram']
    dual = np.concatenate((fpz,pz),axis=1)
    
read_data('SC4001E0')
data_reshape(eeg_fpz,eeg_pz,hyp)


fs=100
epochs =30
encoder = LabelEncoder()
encoder.fit(label)
encoded_y = encoder.transform(label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)
y = dummy_y

classes = 6
model = Sequential()
def baseline_model_single():
    #model.add(Conv1D(filters=128,kernel_size=5,activation='relu',batch_input_shape=(None,3000,1)))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(rate=0.25))
    #model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['accuracy'])
    return model

def baseline_model_dual():
    model.add(Conv1D(filters=128,kernel_size=2,activation='relu',input_shape=(6000,1)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model_single, epochs=200, batch_size=5, verbose=0)
estimator_1 = KerasClassifier(build_fn=baseline_model_dual, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)

results_fpz = cross_val_score(estimator, fpz, y, cv=kfold)
print("Baseline for fpz channel: %.2f%% (%.2f%%)" % (results_fpz.mean()*100, results_fpz.std()*100))

results_pz = cross_val_score(estimator, pz, y, cv=kfold)
print("Baseline for pz channel: %.2f%% (%.2f%%)" % (results_pz.mean()*100, results_pz.std()*100))

results_dual = cross_val_score(estimator_1, dual, y, cv=kfold)
print("Baseline for dual channel: %.2f%% (%.2f%%)" % (results_dual.mean()*100, results_dual.std()*100))
#test_y_predictions = model.predict(x_test_cnn)