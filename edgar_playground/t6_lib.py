##### COPY__PASTE__LIB__BEGIN #####
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import math
import gc
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.layers import Dense, Input, Activation
from keras.layers import BatchNormalization, Add, Dropout
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import callbacks
from keras import backend as K

##### COPY__PASTE__LIB__END #####