# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:54:47 2019

@author: Aadarsh Srivastava
"""
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.layers
from keras import losses
from keras.utils import to_categorical
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
#read in data using pandas
train_df = pd.read_csv(r'C:\Users\Aadarsh Srivastava\OneDrive\Desktop\Project GIthub\Data CSV\file_new.csv')
#check data has been read in properly
train_df.head()

#create a dataframe with all training data except the target column
x = train_df.drop(columns=['hyp'])

#check that the target variable has been removed
x.head()

#create a dataframe with only the target column
y = train_df[['hyp']]
y = to_categorical(y)
#for e in range(1,len(train_y)):
 #   train_y[e] = float(train_y[e])
#view dataframe

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#create model
model = Sequential()

#get number of columns in training data
n_cols = x_train.shape[1]
classes = 7
#add model layers
model.add(keras.layers.Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(classes, activation='softmax'))


#compile model using mse as a measure of model performance
model.compile(optimizer='sgd', loss=losses.categorical_crossentropy)

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)
#train model
model.fit(x_train, y_train, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])
test_y_predictions = model.predict(x_test)
#test_y_predictions=np.round(test_y_predictions)
actual = y_test
predicted =  test_y_predictions
print('Accuracy Score :',accuracy_score(actual, predicted)*100)
print('Report : ')
print(classification_report(actual, predicted)) 