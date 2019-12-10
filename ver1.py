# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 21:54:00 2019

@author: Aadarsh Srivastava
"""
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D,LSTM,SpatialDropout1D,Dropout,Dense
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score

def read_mat(file):
    eeg_fpz  = loadmat('D:/Project/Github_dwn/'+file+'/matlab/eeg_fpz.mat')
    eeg_pz  = loadmat('D:/Project/Github_dwn/'+file+'/matlab/eeg_fpz.mat')
    hyp = loadmat('D:/Project/Github_dwn/'+file+'/matlab/hypnogram.mat')
    l1 = eeg_fpz['signal']
    l2 = eeg_pz['signal']
    l3 = hyp['hypnogram']
    return l1,l2,l3

def merge_data(fpz,pz,label,file):
    temp1,temp2,temp3=read_mat(file)
    fpz=np.concatenate((fpz,temp1),axis=0)
    pz=np.concatenate((pz,temp2),axis=0)
    label=np.concatenate((label,temp3),axis=0)
    return fpz,pz,label

def data_reshape(fpz,pz,label):
    fpz= np.reshape(fpz,(label.shape[0],samples,1))
    pz= np.reshape(pz,(label.shape[0],samples,1))
    return fpz,pz


def model_single():
    model = Sequential()
    model.add(Conv1D(filters=128,kernel_size=2,activation='relu',input_shape=(3000,1)))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(256,dropout=0.3,recurrent_dropout=0.3))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['accuracy'])
    return model
#THE SAMPLING FREQUENCY AND EPOCHS TIME
fs = 100
epochs = 30
samples = fs*epochs

fpz,pz,label = read_mat('SC4001E0')
fpz,pz,label = merge_data(fpz,pz,label,'SC4002E0')
#fpz,pz,label = merge_data(fpz,pz,label,'SC4011E0')
#fpz,pz,label = merge_data(fpz,pz,label,'SC4012E0')
fpz,pz = data_reshape(fpz,pz,label)

encoder = LabelEncoder()
encoder.fit(label)
label = encoder.transform(label)
label = to_categorical(label)

x_train,x_test,y_train,y_test = train_test_split(fpz,label,test_size=0.2,random_state=1)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,random_state=1)

model_fpz = model_single()
model_fpz.fit(x=x_train,y=y_train,epochs=5,verbose=1)#,callbacks=[EarlyStopping(patience=3,restore_best_weights=True)])
pred_fpz = model_fpz.predict(x_test,verbose=1)
label_fpz = np.argmax(pred_fpz,axis=-1)
y_test_label = np.argmax(y_test,axis=-1)
print('Accuracy : {} %'.format(accuracy_score(y_test_label,label_fpz)*100))
