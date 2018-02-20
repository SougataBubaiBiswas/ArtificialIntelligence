import numpy as np
#import theano
import matplotlib
import scipy
import keras
import tensorflow as tf

np.random.seed(111)

# Keras model module
from keras.models import Sequential
# Keras core layers
from keras.layers import Dense, Dropout, Convolution2D, AveragePooling2D, MaxPooling2D, Flatten, Activation
# Keras cnn layer
from keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.callbacks import EarlyStopping
# Utilities
from keras.utils import np_utils
# Load cifar10 data set
from keras.datasets import cifar10
# Load MNIST data set
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
# (60000, 28, 28)
# Plotting first sample of plot train
from matplotlib import pyplot as plt
plt.imshow(X_train[0])
# Reshape input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
print(X_train.shape)
# (60000, 1, 28, 28)
# Convert data type and normalize values
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(y_train.shape)
# (60000,)
print(y_train[:10])
# Preprocess class labelsPython
# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print(Y_train.shape)
# (60000, 10)
### Define Model Architecture
# Declare Sequential model
model = Sequential()
# CNN input layerPython
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(3,28,28)))
print(model.output_shape)
# (None, 32, 26, 26)
# Adding more layers
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Fully connected Dense layersPython
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile modelPython
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


