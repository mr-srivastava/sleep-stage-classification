# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:22:49 2019

@author: Aadarsh Srivastava
"""# first neural network with keras tutorial
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = pd.read_csv(r'C:\Users\Aadarsh Srivastava\OneDrive\Desktop\file_new.csv',skiprows=1)
# split into input (X) and output (y) variables
X = dataset[:,0:3]
y = dataset[:,3]


# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))