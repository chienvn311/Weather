import getData
import numpy as np
import Test.Test as tt
from Model.Models import Learning

#1 - 0/ Rain Tommorow
file_location = 0
# file_location = 1
# train_data = 0
train_data = 3


weather_data = getData.get_data(file_location)
weather_data = getData.clear_data(weather_data)
train, test, train_lb, test_lb = getData.slip_data(weather_data, getData.get_train_lb(train_data))
learning = Learning(train, test, train_lb)


# logistic_prediction, knn_prediction, svm_prediction, rf_prediction = learning.train_model()
# report = np.array([test_lb, logistic_prediction, knn_prediction, svm_prediction, rf_prediction])
# report2 = {
#     'Logistic Regression': tt.mean_absolute(test_lb, logistic_prediction),
#     'KNN': tt.mean_absolute(test_lb, knn_prediction),
#     'Support Vector Machine': tt.mean_absolute(test_lb, svm_prediction),
#     'Random Forest': tt.mean_absolute(test_lb, rf_prediction)
# }

# xbo = learning.train_single()
# report = np.array([test_lb, xbo])

lstm = learning.train_lstm()
report = np.array([test_lb, lstm])




