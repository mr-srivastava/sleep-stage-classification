# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 11:13:47 2020

@author: Aadarsh Srivastava
"""

from mne.io import read_raw_edf
from mne import read_annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D,LSTM,MaxPooling1D,AveragePooling1D,SpatialDropout1D,Dropout,Dense
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


def hyp_reader(data):
    desc=data.description
    dur=data.duration
    dur=dur.astype(int)
    def rep(index):
        temp = np.array([desc[index]]*(int(dur[index]/30)))
        return temp
    hyp = rep(0)
    for i in range(1,len(dur)-1):
        hyp=np.concatenate((hyp,rep(i)),axis=0)
    return hyp
    

def read_EDF(psg,hyp):
    data  = read_raw_edf('D:/Project/Dataset/Sleep EDF Database Expanded/sleep-cassette/'+psg+'.edf')
    raw_data = data.get_data()
    data_hyp  = read_annotations('D:/Project/Dataset/Sleep EDF Database Expanded/sleep-cassette/'+hyp+'.edf')
    l1 = raw_data[0]
    l2 = raw_data[1]
    l3 = hyp_reader(data_hyp)
    return l1,l2,l3

def merge_data(fpz,pz,label,psg,hyp):
    temp1,temp2,temp3=read_EDF(psg,hyp)
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
    model.add(Conv1D(filters=128,kernel_size=8,activation='relu',input_shape=(3000,1)))
    model.add(Conv1D(filters=64,kernel_size=2,activation='relu',input_shape=(128,1)))
    model.add(AveragePooling1D(pool_size=2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(256,dropout=0.3,recurrent_dropout=0.3))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['accuracy'])
    return model


fs = 100
epochs = 30
samples = fs*epochs

fpz,pz,label = read_EDF('SC4001E0-PSG','SC4001EC-Hypnogram')
fpz,pz,label = merge_data(fpz,pz,label,'SC4002E0-PSG','SC4002EC-Hypnogram')
fpz,pz,label = merge_data(fpz,pz,label,'SC4062E0-PSG','SC4062EC-Hypnogram')
fpz,pz,label = merge_data(fpz,pz,label,'SC4031E0-PSG','SC4031EC-Hypnogram')
fpz,pz,label = merge_data(fpz,pz,label,'SC4041E0-PSG','SC4041EC-Hypnogram')
fpz,pz,label = merge_data(fpz,pz,label,'SC4042E0-PSG','SC4042EC-Hypnogram')
fpz,pz,label = merge_data(fpz,pz,label,'SC4051E0-PSG','SC4051EC-Hypnogram')
fpz,pz,label = merge_data(fpz,pz,label,'SC4052E0-PSG','SC4052EC-Hypnogram')

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
