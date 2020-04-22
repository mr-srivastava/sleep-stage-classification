import os
import glob
import numpy as np
import pandas as pd
from Preprocess import reader,join_data,spectrogram
from keras_preprocessing.image import ImageDataGenerator

#Uncomment before running
#Enter the root directory and image directory
'''
root_dir='C:/Users/Aadarsh Srivastava/OneDrive/Desktop/Project Github/Data'
img_dir='C:/Users/Aadarsh Srivastava/OneDrive/Desktop/Project Github/Trial/Spectrogram'
'''

#Read .npz data of Sleep and its label
files = glob.glob(os.path.join(root_dir,"*.npz"))


#Loop over all the files
for i in range(len(files)):
    if i==0:
        signal,stages=reader(i,files)
    else:
        signal,stages=join_data(signal,stages,i,files)

#Reshape the signal files        
signal_plot=np.reshape(signal,(signal.shape[0],signal.shape[1]))

#Loop over signals to plot spectrogram
for i in range(len(stages)):
    spectrogram(i,signal_plot,img_dir)


label_df=pd.DataFrame(data=stages,index=range(len(stages)))
label_df['Index']=label_df.index
label_df.columns=["Label","Index"]

def append_name(fn):
    return 'spect_img'+str(fn)+'.png'

def label_convert(fn):
    return "Sleep Stage "+str(fn)

label_df["Index"]=label_df["Index"].apply(append_name)
label_df["Label"]=label_df["Label"].apply(label_convert)
img_files = glob.glob(os.path.join(img_dir,"*.png"))
label=label_df.head(len(img_files))


datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.20)
traindata=datagen.flow_from_dataframe(label, directory=img_dir,x_col="Index",y_col="Label",subset="training",target_size=(32,32))
testdata=datagen.flow_from_dataframe(label, directory=img_dir,x_col="Index",y_col="Label",subset="validation",target_size=(32,32))

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras import regularizers,optimizers
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

STEP_SIZE_TRAIN=traindata.n//traindata.batch_size
STEP_SIZE_TEST=testdata.n//testdata.batch_size

model.fit_generator(generator=traindata,steps_per_epoch=STEP_SIZE_TRAIN,epochs=10)
