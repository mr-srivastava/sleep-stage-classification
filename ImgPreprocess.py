import numpy as np
import matplotlib.pyplot as plt

#Function to read .npz files
def reader(i,files):
    file=np.load(files[i])
    return file['x'],file['y']

#Function to join data of all .npz files together
def join_data(x,y,i,files):
    temp_x,temp_y=reader(i,files)
    x=np.concatenate((x,temp_x),axis=0)
    y=np.concatenate((y,temp_y),axis=0)
    return x,y

#Function to create Spectrogram and save it in Image directory
def spectrogram(i,signal,img_dir):
    plt.specgram(signal[i],Fs=100)
    plt.axis('off')
    plt.savefig(img_dir+'/spect_img'+str(i)+'.png',bbox_inches='tight',pad_inches=0)