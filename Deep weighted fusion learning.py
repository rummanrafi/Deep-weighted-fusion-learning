#!/usr/bin/env python
# coding: utf-8

# In[2]:


# [1] Image sensor data 

import os
import pywt
import copy
import glob
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical

#training_data
DATADIR = 'C:/PhD/Semesters/Spring 2022/Research/SensorFusion/Paper/Database/Ours_data/training/image'
CATEGORIES = ['Zero', 'One', 'Two']

training_data=[]
IMG_SIZE=50

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
X_train = []
y_train = []
for features, labels in training_data:
    X_train.append(features)
    y_train.append(labels)

X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_train_image = X_train / 255.0
y_train_image = to_categorical(y_train)

#testing_data
DATADIR = 'C:/PhD/Semesters/Spring 2022/Research/SensorFusion/Paper/Database/Ours_data/testing/image'
CATEGORIES = ['Zero', 'One', 'Two']

testing_data=[]
IMG_SIZE=50

def create_testing_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                testing_data.append([new_array, class_num])
            except Exception as e:
                pass

create_testing_data()
X_test = []
y_test = []
for features, labels in testing_data:
    X_test.append(features)
    y_test.append(labels)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_test_image = X_test / 255.0
y_test_image = to_categorical(y_test)


# CNN model for feature extraction
activation = 'sigmoid'

feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (50, 50, 1)))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Flatten())
#Add layers for deep learning prediction
x = feature_extractor.output  
x = Dense(128, activation = activation, kernel_initializer = 'he_uniform')(x)
prediction_layer = Dense(3, activation = 'softmax')(x)
cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
cnn_model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = cnn_model.fit(X_train_image, y_train_image, epochs=20, validation_data = (X_test_image, y_test_image))

#loss curve
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#accuracy curve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#extracted features for training data
X_train_cnn = feature_extractor.predict(X_train_image)
print('Extracted features from training image data: ', X_train_cnn)

#extracted features for testing data
X_test_cnn = feature_extractor.predict(X_test_image)
print('Extracted features from testing image data: ', X_test_cnn)




# [2] CO2 sensor data

#training_data
df = pd.read_csv('C:/PhD/Semesters/Spring 2022/Research/SensorFusion/Paper/Database/Ours_data/training/training_baseline.csv')
composite_signal =  df['CO2'].values
composite_signal
def filter_bank(index_list, wavefunc='db4', lv=4, m=1, n=4, plot=False):
    
    '''
    WT: Wavelet Transformation Function
    index_list: Input Sequence;
   
    lv: Decomposing Level；
 
    wavefunc: Function of Wavelet, 'db4' default；
    
    m, n: Level of Threshold Processing
   
    '''
   
    # Decomposing 
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   #  Decomposing by levels，cD is the details coefficient
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn function
# Denoising
    # Soft Threshold Processing Method
    for i in range(m,n+1):   #  Select m~n Levels of the wavelet coefficients，and no need to dispose the cA coefficients(approximation coefficients)
        cD = coeff[i]
        Tr = np.sqrt(2*np.log2(len(cD)))  # Compute Threshold
        for j in range(len(cD)):
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) * (np.abs(cD[j]) -  Tr)  # Shrink to zero
            else:
                coeff[i][j] = 0   # Set to zero if smaller than threshold
# Reconstructing
    coeffs = {}
    for i in range(len(coeff)):
        coeffs[i] = copy.deepcopy(coeff)
        for j in range(len(coeff)):
            if j != i:
                coeffs[i][j] = np.zeros_like(coeff[j])
    
    for i in range(len(coeff)):
        coeff[i] = pywt.waverec(coeffs[i], wavefunc)
        if len(coeff[i]) > len(index_list):
            coeff[i] = coeff[i][:-1]
        
    if plot:     
        denoised_index = np.sum(coeff, axis=0)   
        data = pd.DataFrame({'CLOSE': index_list, 'denoised': denoised_index})
        data.plot(figsize=(10,10),subplots=(2,1))
        data.plot(figsize=(10,5))
   
    return coeff
coeff=filter_bank(composite_signal,plot=True)
fig, ax =  plt.subplots(len(coeff), 1, figsize=(10, 20))
for i in range(len(coeff)):
    if i == 0:
        ax[i].plot(coeff[i], label = 'cA[%.0f]'%(len(coeff)-i-1))
        ax[i].legend(loc = 'best')
    else:
        ax[i].plot(coeff[i], label = 'cD[%.0f]'%(len(coeff)-i))
        ax[i].legend(loc = 'best')
        
X_CO2 = coeff[0]
print('Features extracted from CO2 sensor data:', X_CO2)
df['CO2'] = X_CO2




# [3] PIR sensor data

# training_data
df = pd.read_csv('C:/PhD/Semesters/Spring 2022/Research/SensorFusion/Paper/Database/Ours_data/training/training_baseline.csv')
composite_signal =  df['PIR'].values
composite_signal
def filter_bank(index_list, wavefunc='db4', lv=4, m=1, n=4, plot=False):
    
    '''
    WT: Wavelet Transformation Function
    index_list: Input Sequence;
   
    lv: Decomposing Level；
 
    wavefunc: Function of Wavelet, 'db4' default；
    
    m, n: Level of Threshold Processing
   
    '''
   
    # Decomposing 
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   #  Decomposing by levels，cD is the details coefficient
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn function
# Denoising
    # Soft Threshold Processing Method
    for i in range(m,n+1):   #  Select m~n Levels of the wavelet coefficients，and no need to dispose the cA coefficients(approximation coefficients)
        cD = coeff[i]
        Tr = np.sqrt(2*np.log2(len(cD)))  # Compute Threshold
        for j in range(len(cD)):
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) * (np.abs(cD[j]) -  Tr)  # Shrink to zero
            else:
                coeff[i][j] = 0   # Set to zero if smaller than threshold
# Reconstructing
    coeffs = {}
    for i in range(len(coeff)):
        coeffs[i] = copy.deepcopy(coeff)
        for j in range(len(coeff)):
            if j != i:
                coeffs[i][j] = np.zeros_like(coeff[j])
    
    for i in range(len(coeff)):
        coeff[i] = pywt.waverec(coeffs[i], wavefunc)
        if len(coeff[i]) > len(index_list):
            coeff[i] = coeff[i][:-1]
        
    if plot:     
        denoised_index = np.sum(coeff, axis=0)   
        data = pd.DataFrame({'CLOSE': index_list, 'denoised': denoised_index})
        data.plot(figsize=(10,10),subplots=(2,1))
        data.plot(figsize=(10,5))
   
    return coeff
coeff=filter_bank(composite_signal,plot=True)
fig, ax =  plt.subplots(len(coeff), 1, figsize=(10, 20))
for i in range(len(coeff)):
    if i == 0:
        ax[i].plot(coeff[i], label = 'cA[%.0f]'%(len(coeff)-i-1))
        ax[i].legend(loc = 'best')
    else:
        ax[i].plot(coeff[i], label = 'cD[%.0f]'%(len(coeff)-i))
        ax[i].legend(loc = 'best')
        
X_pir = coeff[0]
print('Features extracted from PIR sensor data: ', X_pir)



# [4] Weighted Model
df['PIR'] = X_pir
df['CO2'] = X_CO2
X_train_cp = df.iloc[:,1:3]
y_train_cp = df['Occupancy']
scaler = MinMaxScaler()
X_train_cp = scaler.fit_transform(X_train_cp)
y_train_cp = to_categorical(y_train_cp)

w1=0.5
w2=0.3
w3=0.2
w4=0.6
w5=0.3
w6=0.1

# Intra attention model
intra_weighted_1_image = w1*X_train_cnn 
intra_weighted_1_co2 = w2*X_train_cp[:,0] 
intra_weighted_1_pir = w3*X_train_cp[:,1] 
            
intra_weighted_2_image = w4*X_train_cnn 
intra_weighted_2_co2 = w5*X_train_cp[:,0] 
intra_weighted_2_pir = w6*X_train_cp[:,1] 

reweighted_image = np.dot(np.transpose(intra_weighted_1_image), intra_weighted_2_image)
reweighted_co2 = np.dot(np.transpose(intra_weighted_1_co2), intra_weighted_2_co2)
reweighted_pir = np.dot(np.transpose(intra_weighted_1_pir), intra_weighted_2_pir)

self_weighted_image = keras.activations.sigmoid(reweighted_image)
self_weighted_co2 = keras.activations.sigmoid(reweighted_co2)
self_weighted_pir = keras.activations.sigmoid(reweighted_pir)

intra_weighted_image = np.dot(self_weighted_image, np.transpose(X_train_cnn))
intra_weighted_image = intra_weighted_image / 10000
intra_weighted_co2 = np.dot(self_weighted_co2, np.transpose(X_train_cp[:,0]))
intra_weighted_pir = np.dot(self_weighted_pir, np.transpose(X_train_cp[:,1]))

# Cross attention model
w1_IC = 0.5
w2_CP = 0.2
w3_PI = 0.3
w4_IC = 0.6
w5_CP = 0.1
w6_PI = 0.3

con_IC = intra_weighted_image + intra_weighted_co2
con_CP = intra_weighted_co2 + intra_weighted_pir
con_PI = intra_weighted_pir + intra_weighted_image

inter_weighted_IC_1 = w1_IC*con_IC
inter_weighted_CP_1 = w2_CP*con_CP
inter_weighted_PI_1 = w3_PI*con_PI

inter_weighted_IC_2 = w4_IC*con_IC
inter_weighted_CP_2 = w5_CP*con_CP
inter_weighted_PI_2 = w6_PI*con_PI

cross_weighted_IC = np.dot(np.transpose(inter_weighted_IC_1), inter_weighted_IC_2)
cross_weighted_CP = np.dot(np.transpose(inter_weighted_CP_1), inter_weighted_CP_2)
cross_weighted_PI = np.dot(np.transpose(inter_weighted_PI_1), inter_weighted_PI_2)

activated_weighted_IC = keras.activations.sigmoid(cross_weighted_IC)
activated_weighted_CP = keras.activations.sigmoid(cross_weighted_CP)
activated_weighted_PI = keras.activations.sigmoid(cross_weighted_PI)

inter_weighted_IC = activated_weighted_IC*intra_weighted_pir
inter_weighted_CP = np.dot(activated_weighted_CP, np.transpose(intra_weighted_image))
inter_weighted_PI = activated_weighted_PI*intra_weighted_co2

weighted_data = np.concatenate((inter_weighted_IC, inter_weighted_CP, inter_weighted_PI), axis=1)

from sklearn.model_selection import train_test_split
y = y_train_image

trainX, testX, trainY, testY = train_test_split(weighted_data, y, test_size=0.33, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(9878,), activation='relu'),
    keras.layers.Dense(32,  activation='relu'),
    keras.layers.Dense(8,  activation='relu'),
    keras.layers.Dense(3,  activation='softmax')
])


opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

history = model.fit(trainX, trainY, epochs=200, validation_data = (testX, testY))

#loss curve
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.rcParams['font.size'] = '16'
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#accuracy curve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




