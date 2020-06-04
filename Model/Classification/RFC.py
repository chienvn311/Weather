import Model.Classification.Input.get_input as input
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# dir = "/Users/chienvn/PycharmProjects/Weather/Files/weather_include.csv"
# weather_data = pds.read_csv(dir, header=0)
# weather_data = pds.get_dummies(weather_data.drop('Date', axis=1))
# weather_data = weather_data.drop('Rain_today_No', axis=1)
# weather_data = weather_data.drop('Rain_Tomorrow_No', axis=1)
# weather_data = weather_data.drop('Rainfall_Tomorrow', axis=1)
# dependence = np.array(weather_data['Rain_Tomorrow_Yes'])
# weather_data = weather_data.drop('Rain_Tomorrow_Yes', axis=1)
# # weather_data = weather_data.drop('Time_of_maximum_wind_gust', axis=1)
# independence = np.array(weather_data)
# train, test, train_lb, test_lb = train_test_split(independence, dependence, test_size=0.25, random_state=42)


dir = "/Users/chienvn/PycharmProjects/Weather/Model/Classification/Input/new_weather_Ex.csv"
train, test, train_lb, test_lb = input.get_input_new(dir)


rf_classification = RandomForestClassifier(n_estimators=100)
rf_classification.fit(train, train_lb)
prediction = rf_classification.predict(test)
print(confusion_matrix(test_lb, prediction))
print(classification_report(test_lb, prediction))


dir_test = "/Users/chienvn/PycharmProjects/Weather/Model/Classification/Input/new_weather_Test.csv"
new_test, new_test_lb = input.get_input_test(dir_test)
new_prediction = rf_classification.predict(new_test)
print(confusion_matrix(new_test_lb, new_prediction))
print(classification_report(new_test_lb, new_prediction))

