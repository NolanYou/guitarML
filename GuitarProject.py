#TODO: audio input, debugging after
#TODO: pandas for audio input maybe?
#Guide source https://www.section.io/engineering-education/machine-learning-for-audio-classification/


import os
import librosa
import librosa.display
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import IPython.display
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import IPython.display as ipd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
# %matplotlib inline

file_name='guitarml.wav'

audio_data, sampling_rate = librosa.load(file_name)



def make_normalize_mfccs(file_string):

    audio_data, sample_rate = librosa.load(file_string, res_type='kaiser_fast')
    #kaiser_fast is a resampling type that is optimized towards speed as opposed to kaiser_best

    # todo: play around with n_mfccs, ignore averaging to make 2d and play around with convolutions

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=50)
    #to summarize, we are taking 50 mfcc samples (slices) of our audio
    mfccs_avged = np.mean(mfccs.T, axis=0)
    print(np.shape(mfccs.T))
    #axis = 0 tells us we are averaging down the rows
    return mfccs_avged

make_normalize_mfccs(file_name)


def get_data(folder_path):
    #todo: implement
    mfccs = np.array()
    y_data_noncat = []
    distort_folder_audios = None
    reverb_folder_audios = None
    clean_folder_audios = None

    for i in distort_folder_audios:
        mfccs.append(make_normalize_mfccs(i))
        y_data_noncat.append("")
        
    for j in reverb_folder_audios:
        mfccs.append(make_normalize_mfccs(j))
        y_data_noncat.append(1)
    
    for k in clean_folder_audios():
        mfccs.append(make_normalize_mfccs(k))
        y_data_noncat.append(2)
        
    return  mfccs, to_categorical(y_data_noncat)

def run():
    X_train, X_test, Y_train, Y_test = train_test_split(get_data(""))

    model = Sequential()

    model.add(Dense(100, activation="relu", input_shape = (50,)))
    model.add(Dropout(0.1))

    model.add(Dense(400, activation= "relu", input_shape = (50,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(400, activation="relu", input_shape=(50,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(3), activation = "softmax")

    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

    num_epochs = 150
    num_batch_size = 24


    save_location = ""
    #save location should be a file in hdf5
    model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test),
              callbacks=[save_location], verbose=1)

    test_accuracy = model.evaluate(X_test, Y_test, verbose=0)





