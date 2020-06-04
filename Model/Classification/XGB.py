import Model.Classification.Input.get_input as input
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


# dir = "/Users/chienvn/PycharmProjects/Weather/Files/weather_exclude.csv"
# dir = "/Users/chienvn/PycharmProjects/Weather/Files/weather_include.csv"
dir = "/Users/chienvn/PycharmProjects/Weather/Model/Classification/Input/new_weather_Ex.csv"
train, test, train_lb, test_lb = input.get_input_new(dir)

xg_classification = XGBClassifier(max_depth=10)
xg_classification.fit(train, train_lb)
prediction = xg_classification.predict(test)
print(confusion_matrix(test_lb, prediction))
print(classification_report(test_lb, prediction))


dir_test = "/Users/chienvn/PycharmProjects/Weather/Model/Classification/Input/new_weather_Test.csv"
new_test, new_test_lb = input.get_input_test(dir_test)
new_prediction = xg_classification.predict(new_test)
print(confusion_matrix(new_test_lb, new_prediction))
print(classification_report(new_test_lb, new_prediction))
