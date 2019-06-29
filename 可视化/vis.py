seed_value = 0
import matplotlib.pyplot as plt
from keras.models import load_model
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.set_random_seed(seed_value)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(),config = session_conf)
K.set_session(sess)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM, Flatten
from keras.layers import AveragePooling1D, MaxPooling1D, Bidirectional, GlobalMaxPool1D, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D,concatenate
from keras.layers import SpatialDropout1D
from keras.models import Sequential
from keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os
from math import floor
import warnings

######################################################################################################
# train data
with open('new_data_filter0.pickle', 'rb') as f:
    Data = pickle.load(f)
    Data = pd.DataFrame(Data)

numtrain = 186 * 6 * 12
numval = 2678

# Train validation split
idx = random.sample(range(numtrain), numval) # 80% for training
all_index = list(np.setdiff1d(range(numtrain), idx))

# Getting X_train, X_val, y_train and scale them

X_train = Data.iloc[all_index,:].drop(2251, axis = 1)
X_val = Data.iloc[idx,:].drop(2251, axis = 1)
y_train = pd.get_dummies(Data.iloc[all_index,:][2251])

names = y_train.columns
mapping = {}
i = 0
for n in names:
    mapping[i] = n
    i+=1

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
######################################################################################################
cnn_model= load_model('lenet_1119_1919_BEST.h5')
Test = pd.read_csv('test_data_19.csv')
X_test = scaler.transform(Test.iloc[:,1:])




with open("dat.txt", "w") as f:
    for i in range(100):
        for t in range(2251):
            f.write(str(X_test[0, t]) + ' ')
        f.write("\n")

quit()


######################################################################################################
image_arr=np.reshape(X_test[0],(2251,1))
print(cnn_model.layers[0].input, cnn_model.layers[1].output)

image_arr= image_arr.reshape(-1,2251,1)
print(image_arr.shape)

layer_1 = K.function([cnn_model.layers[0].input], [cnn_model.layers[1].output])
f1 = layer_1([image_arr])[0]

print(f1.shape)

re = np.transpose(f1, (0,2,1))
print(re.shape)
print(re)

np.savetxt('test.out', f1[0], delimiter=',')
######################################################################################################

layer_2 = K.function([model.layers[0].input], [model.layers[7].output])
f2 = layer_2([image_arr])[0]

re = np.transpose(f2, (0,3,1,2))




