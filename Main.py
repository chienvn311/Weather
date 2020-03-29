import getData
import pandas as pd
import numpy as np
from Model.Models import Learning


weather_data = getData.get_data(1)
weather_data = getData.clear_data(weather_data)
train, test, train_lb, test_lb = getData.slip_data(weather_data, getData.get_train_lb(0))
learning = Learning(train, test, train_lb)
# logistic_prediction, knn_prediction, svm_prediction, rf_prediction = learning.train_model()
# report = np.array([test_lb, logistic_prediction, knn_prediction, svm_prediction, rf_prediction])
nn_prediction = learning.train_nn()
report = np.array([test_lb, nn_prediction])

