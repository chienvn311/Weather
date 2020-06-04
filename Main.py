import getData
import numpy as np
import Test.Test as tt
from Model.Models import Learning
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd


#1 - 0/ Rain Tommorow
# file_location = 0
file_location = 1
# 0: "/Users/chienvn/PycharmProjects/Weather/Files/weather_include.csv",
# 1: "/Users/chienvn/PycharmProjects/Weather/Files/weather_exclude.csv",
# 2: "/Users/chienvn/PycharmProjects/Weather/Files/weather_consider.csv",

# train_data = 0
train_data = 0

# 0: "Rain_Tomorrow_Yes",
#         1: "Rain_today_Yes",
#         2: "Rainfall",
#         3: "Rainfall_Tomorrow",

weather_data = getData.get_data(file_location)
weather_data = getData.clear_data(weather_data)
train, test, train_lb, test_lb = getData.slip_data(weather_data, getData.get_train_lb(train_data))
learning = Learning(train, test, train_lb)

# new = pd.DataFrame(train)
# new_train = new[[3, 4, 5, 7, 10, 11, 12, 15]].to_numpy()
# t = pd.DataFrame(test)
# new_test = t[[3, 4, 5, 7, 10, 11, 12, 15]].to_numpy()
# learning = Learning(new_train, new_test, train_lb)

logistic_prediction, knn_prediction, svm_prediction, rf_prediction = learning.train_model()
report = np.array([test_lb, logistic_prediction, knn_prediction, svm_prediction, rf_prediction])
report2 = {
    'Logistic Regression': tt.mean_absolute(test_lb, logistic_prediction),
    'KNN': tt.mean_absolute(test_lb, knn_prediction),
    'Support Vector Machine': tt.mean_absolute(test_lb, svm_prediction),
    'Random Forest': tt.mean_absolute(test_lb, rf_prediction)
}

# xbo = learning.train_single()
# report = np.array([test_lb, xbo])

# nn_prediction = learning.train_nn()
# report = np.array([test_lb, nn_prediction.reshape(-1)])
#
# mean_squared_error(test_lb, logistic_prediction)
# mean_absolute_error(test_lb, logistic_prediction)
#
# df = pd.DataFrame({'Actual': test_lb, 'Predicted': xbo})
# df.plot(kind='line',figsize=(10, 8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='blue')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='orange')
# plt.show()

# df = pd.DataFrame({'Actual': test_lb, 'Predicted': logistic_prediction})
# df.plot(kind='line',figsize=(10, 8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='blue')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='orange')
# plt.show()
#
# df2 = pd.DataFrame({'Actual': test_lb, 'Predicted': knn_prediction})
# df2.plot(kind='line',figsize=(10,8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='blue')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='orange')
# plt.show()
#
# df3 = pd.DataFrame({'Actual': test_lb, 'Predicted': svm_prediction})
# df3.plot(kind='line',figsize=(10,8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='blue')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='orange')
# plt.show()
#
# df4 = pd.DataFrame({'Actual': test_lb, 'Predicted': rf_prediction})
# df4.plot(kind='line',figsize=(10,8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='blue')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='orange')
# plt.show()
# getData.export_numpy_csv(report)