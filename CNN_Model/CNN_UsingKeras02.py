# Source : https://www.kernix.com/blog/a-toy-convolutional-neural-network-for-image-classification-with-keras_p14
from __future__ import print_function

import os
import re

import numpy as np
import PIL
from PIL import Image

from scipy.stats import randint as sp_randint

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, AveragePooling2D, MaxPooling2D, Flatten, Activation
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.datasets import cifar10

%matplotlib inline