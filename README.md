# Sleep Stage Classification 
Currently, sleep disorders are considered as one of the major human life issues. There are several stable physiological stages that the human brain goes through during sleep. This project deals with automatic classification of the sleep stage based on the EEG Signals recorded.


# Dataset
The dataset used for the project is the [Sleep-EDF database](https://archive.physionet.org/physiobank/database/sleep-edfx/?C=M;O=A) contains 197 whole-night PolySomnoGraphic sleep recordings, containing EEG, EOG, chin EMG, and event markers. Some records also contain respiration and body temperature. Corresponding hypnograms (sleep patterns) were manually scored by well-trained technicians according to the Rechtschaffen and Kales manual, and are also available.

# Approach
The basic approach for the project is to classify the various sleep stages using the EEG signals the help of deep neural networks.
The approaches include using a Conv-LSTM Network,CNN and RNN.
Further using the EEG signals to construct spectrograms and then applying image classification using neural networks is also tried out

# Note 
This this project is part of of the Final Year Project for Bachelor's of Engineering course in Electronics and Communication Engineering specialization for Birla Institute of Technology,Mesra.
