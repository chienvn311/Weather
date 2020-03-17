from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import getData


weather_data = getData.get_data()
train, test, train_lb, test_lb = getData.slip_data(weather_data)



model = LinearRegression()
model.fit(train, train_lb)
prediction = model.predict(test)
print(prediction)
print(test_lb)
print(mean_squared_error(test_lb, prediction))