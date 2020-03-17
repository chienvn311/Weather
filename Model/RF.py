import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import getData


weather_data = getData.get_data()
train, test, train_lb, test_lb = getData.slip_data(weather_data)


rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train, train_lb)
prediction = rf.predict(test)
# error = abs(prediction - test_lb)
#
#
# mape = 100 * (error/test_lb)
# accuracy = 100 - np.mean(mape)
# print(mape)
# print(round(accuracy, 2))


print(prediction)
print(test_lb)
print(mean_squared_error(test_lb, prediction))