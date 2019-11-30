# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:54:47 2019

@author: Aadarsh Srivastava
"""
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv1D,Flatten,Dropout,Dense
from keras import losses
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score
#read in data using pandas
train_df = pd.read_csv(r'C:\Users\Aadarsh Srivastava\OneDrive\Desktop\Project Github\Data CSV\file_new.csv')
#check data has been read in properly
train_df.head()

#create a dataframe with all training data except the target column
x = train_df.drop(columns=['hyp'])

#check that the target variable has been removed
x.head()

#create a dataframe with only the target column
y = train_df[['hyp']]
y = pd.DataFrame(to_categorical(y))
y = y.drop(y.columns[0],axis=1)
y = y.rename_axis('ID').values
#for e in range(1,len(train_y)):
 #   train_y[e] = float(train_y[e])
#view dataframe

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)

#Reshape Data
x_train_cnn = x_train.rename_axis('ID').values
x_train_cnn = np.reshape(x_train_cnn,(x_train_cnn.shape[0],x_train_cnn.shape[1],1))
x_test_cnn = x_test.rename_axis('ID').values
x_test_cnn = np.reshape(x_test_cnn,(x_test_cnn.shape[0],x_test_cnn.shape[1],1))
#create model
model = Sequential()

#get number of columns in training data
n_cols = x_train_cnn.shape[1]
classes = 7
#add model layers
model.add(Conv1D(filters=128,kernel_size=2,activation='relu',input_shape=(3,1)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(classes, activation='softmax'))


#compile model using mse as a measure of model performance
model.compile(optimizer='rmsprop', loss=losses.categorical_crossentropy,metrics=['accuracy'])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)
#train model
model.fit(x_train_cnn, y_train, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])
test_y_predictions = model.predict(x_test_cnn)
#test_y_predictions=np.round(test_y_predictions)
actual = y_test
predicted =  test_y_predictions