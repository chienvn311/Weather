import Model.Classification.Input.get_input as input
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


# dir = "/Users/chienvn/PycharmProjects/Weather/Files/weather_exclude.csv"
dir = "/Users/chienvn/PycharmProjects/Weather/Model/Classification/Input/new_weather_Ex.csv"
train, test, train_lb, test_lb = input.get_input_new(dir)


# logistic_regression = LogisticRegression(solver='liblinear', max_iter=3000, random_state=42)
logistic_regression = LogisticRegression(solver='lbfgs', max_iter=3000, random_state=42)
# logistic_regression = LogisticRegression(solver='sag', max_iter=3000, random_state=42)
# logistic_regression = LogisticRegression(solver='newton-cg', max_iter=3000, random_state=42)
logistic_regression.fit(train, train_lb)
prediction = logistic_regression.predict(test)
print(confusion_matrix(test_lb, prediction))
print(classification_report(test_lb, prediction))


dir_test = "/Users/chienvn/PycharmProjects/Weather/Model/Classification/Input/new_weather_Test.csv"
new_test, new_test_lb = input.get_input_test(dir_test)
new_prediction = logistic_regression.predict(new_test)
print(confusion_matrix(new_test_lb, new_prediction))
print(classification_report(new_test_lb, new_prediction))