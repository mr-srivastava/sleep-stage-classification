# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 11:13:47 2020

@author: Aadarsh Srivastava
"""
import numpy as np
import Data_Prep as dp
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import Conv1D,LSTM,MaxPooling1D,AveragePooling1D,SpatialDropout1D,Dropout,Dense,Input
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
from keras.layers.merge import concatenate
from keras.wrappers.scikit_learn import KerasClassifier

def model_single():
    model = Sequential()
    model.add(Conv1D(filters=1500,kernel_size=8,activation='relu',input_shape=(3000,2)))
    model.add(Conv1D(filters=750,kernel_size=2,activation='relu',input_shape=(128,2)))
    model.add(AveragePooling1D(pool_size=2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(256,dropout=0.3,recurrent_dropout=0.3))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['accuracy'])
    return model

def model_trial():
    x = Input(shape=(12000,1))
    y1 = Conv1D(filters=64,kernel_size=50,activation='relu')(x)
    y2 = MaxPooling1D(pool_size=8)(y1)
    y3 = Dropout(0.5)(y2)
    y4 = Conv1D(filters=128,kernel_size=8,activation='relu')(y3)
    y5 = Conv1D(filters=128,kernel_size=8,activation='relu')(y4)
    y6 = Conv1D(filters=128,kernel_size=8,activation='relu')(y5)
    y7 = MaxPooling1D(pool_size=4)(y6)
    
    z1 = Conv1D(filters=64,kernel_size=50,activation='relu')(x)
    z2 = MaxPooling1D(pool_size=8)(z1)
    z3 = Dropout(0.5)(z2)
    z4 = Conv1D(filters=128,kernel_size=8,activation='relu')(z3)
    z5 = Conv1D(filters=128,kernel_size=8,activation='relu')(z4)
    z6 = Conv1D(filters=128,kernel_size=8,activation='relu')(z5)
    z7 = MaxPooling1D(pool_size=4)(z6)
    
    merged = concatenate([y7,z7])
    layer1 = Dropout(0.5)(merged)
    layer2 = LSTM(512,return_sequences=True)(layer1)
    layer3 = Dropout(0.5)(layer2)
    layer4 = LSTM(512)(layer3)
    layer5 = Dropout(0.5)(layer4)
    output = Dense(5, activation='softmax')(layer5)
    
    model = Model(inputs=x,outputs=output)
    print(model.summary)
    model.compile(optimizer='sgd',loss=categorical_crossentropy,metrics=['accuracy'])
    return model


fpz,pz,eog,emg,label = dp.read_EDF('SC4001E0-PSG','SC4001EC-Hypnogram')
fpz,pz,eog,emg,label = dp.merge_data(fpz,pz,eog,emg,label,'SC4002E0-PSG','SC4002EC-Hypnogram')
fpz,pz,eog,emg,label = dp.merge_data(fpz,pz,eog,emg,label,'SC4062E0-PSG','SC4062EC-Hypnogram')
fpz,pz,eog,emg,label = dp.merge_data(fpz,pz,eog,emg,label,'SC4031E0-PSG','SC4031EC-Hypnogram')
fpz,pz,eog,emg,label = dp.merge_data(fpz,pz,eog,emg,label,'SC4041E0-PSG','SC4041EC-Hypnogram')
fpz,pz,eog,emg,label = dp.merge_data(fpz,pz,eog,emg,label,'SC4042E0-PSG','SC4042EC-Hypnogram')
fpz,pz,eog,emg,label = dp.merge_data(fpz,pz,eog,emg,label,'SC4051E0-PSG','SC4051EC-Hypnogram')
fpz,pz,eog,emg,label = dp.merge_data(fpz,pz,eog,emg,label,'SC4052E0-PSG','SC4052EC-Hypnogram')

fpz,pz,eog,emg = dp.data_reshape(fpz,pz,eog,emg,label)
data = np.concatenate((fpz,pz,eog,emg),axis=1)

for i in range(len(label)):
    if(label[i]=='Movement time'):
        label[i]='Sleep stage W'
    elif(label[i]=='Sleep stage 4'):
        label[i]='Sleep stage 3'
    else:
        label[i]=label[i]

trial = KerasClassifier(build_fn=model_trial(),epochs=5,verbose=1)    

encoder = LabelEncoder()
encoder.fit(label)
label = encoder.transform(label)
label = to_categorical(label)


x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.2,random_state=1)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,random_state=1)
model = model_trial()
weights= {0:15250,1:1099,2:3492,3:1450,4:729}
model.fit(x=x_train,y=y_train,epochs=5,verbose=1,class_weight=weights)#,callbacks=[EarlyStopping(patience=3,restore_best_weights=True)])
pred = model.predict(x_test,verbose=1)
label_data = np.argmax(pred,axis=-1)
y_test_label = np.argmax(y_test,axis=-1)
print('Accuracy : {} %'.format(accuracy_score(y_test_label,label_data)*100))

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
def multiclass_roc_auc(y_test,y_pred,average='macro'):
    lb=LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test,y_pred,average=average)
score = multiclass_roc_auc(y_test_label,label_data)
print(score)

