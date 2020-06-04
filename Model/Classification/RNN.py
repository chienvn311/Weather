from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import Model.Classification.Input.get_input as input

dir_test = "/Users/chienvn/PycharmProjects/Weather/Model/Classification/Input/new_weather_Test.csv"
dir_train = "/Users/chienvn/PycharmProjects/Weather/Model/Classification/Input/new_weather_Ex.csv"
train, train_lb = input.get_input_test(dir_train)
test, test_lb = input.get_input_test(dir_test)

encoder = LabelEncoder()

model = Sequential()
model.add(Dense(20, input_dim=60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


