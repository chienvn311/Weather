from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import getData


weather_data = getData.get_data()
train, test, train_lb, test_lb = getData.slip_data(weather_data)


model = SVR(kernel='rbf')
model.fit(train, train_lb)
prediction = model.predict(test)


print(prediction)
print(test_lb)
print(mean_squared_error(test_lb, prediction))