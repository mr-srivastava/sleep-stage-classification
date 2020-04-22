# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:57:13 2019

@author: Aadarsh Srivastava
"""
from keras.models import Sequential
from keras.layers import Conv1D,LSTM,SpatialDropout1D,Dropout,Dense
from keras.losses import categorical_crossentropy




def model_single():
    model = Sequential()
    model.add(Conv1D(filters=128,kernel_size=2,activation='relu',input_shape=(3000,1)))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(256,dropout=0.3,recurrent_dropout=0.3))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['accuracy'])
    return model

def model_dual():
    model = Sequential()
    model.add(Conv1D(filters=128,kernel_size=2,activation='relu',input_shape=(6000,1)))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(256,dropout=0.3,recurrent_dropout=0.3))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['accuracy'])
    return model
