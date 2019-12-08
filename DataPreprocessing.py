# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:57:12 2019

@author: Aadarsh Srivastava
"""
#IMPORT LIBRARIES
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

#THE SAMPLING FREQUENCY AND EPOCHS TIME
fs = 100
epochs = 30
samples = fs*epochs

#THIS FUNCTION READS THE MATLAB FILES OF THE EEG SIGNAL AND HYPNOGRAM FILE
def read_mat(file):
    eeg_fpz  = loadmat('D:/Project/Github_dwn/'+file+'/matlab/eeg_fpz.mat')
    eeg_pz  = loadmat('D:/Project/Github_dwn/'+file+'/matlab/eeg_fpz.mat')
    hyp = loadmat('D:/Project/Github_dwn/'+file+'/matlab/hypnogram.mat')
    return eeg_fpz,eeg_pz,hyp

#THIS FUNCTION RESHAPES THE DATA FOR THE NEURAL NETWORK LAYERS
def data_reshape(file_fpz,file_pz,file_hyp):
    label= file_hyp['hypnogram']
    fpz= file_fpz['signal']
    fpz= np.reshape(fpz,(label.shape[0],samples,1))
    pz= file_pz['signal']
    pz= np.reshape(pz,(label.shape[0],samples,1))
    dual= np.concatenate((fpz,pz),axis=1)
    for i in range(len(label)):
        if(label[i]=='M'):
            label[i]=='W'
    encoder = LabelEncoder()
    encoder.fit(label)
    label = encoder.transform(label)
    # convert integers to dummy variables (i.e. one hot encoded)
    y = to_categorical(label)
    return label,fpz,pz,dual,y


