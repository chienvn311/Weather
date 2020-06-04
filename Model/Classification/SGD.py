import Model.Classification.Input.get_input as input
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier


# dir = "/Users/chienvn/PycharmProjects/Weather/Files/weather_exclude.csv"
dir = "/Users/chienvn/PycharmProjects/Weather/Files/weather_include.csv"
dir = "/Users/chienvn/PycharmProjects/Weather/Model/Classification/Input/new_weather_Ex.csv"
train, test, train_lb, test_lb = input.get_input_new(dir)


SGD_classification = SGDClassifier(shuffle=True, loss='log')
# SGD_classification = SGDClassifier(loss='hinge')
# SGD_classification = SGDClassifier(shuffle=True, loss='modified_huber')
# SGD_classification = SGDClassifier(loss='squared_hinge', random_state=42)
SGD_classification.fit(train, train_lb)
prediction = SGD_classification.predict(test)
print(confusion_matrix(test_lb, prediction))
print(classification_report(test_lb, prediction))


dir_test = "/Users/chienvn/PycharmProjects/Weather/Model/Classification/Input/new_weather_Test.csv"
new_test, new_test_lb = input.get_input_test(dir_test)
new_prediction = SGD_classification.predict(new_test)
print(confusion_matrix(new_test_lb, new_prediction))
print(classification_report(new_test_lb, new_prediction))