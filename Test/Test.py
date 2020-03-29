from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def mean_absolute(actual, prediction):
    return mean_absolute_error(actual, prediction)


def mean_square(actual, prediction):
    return mean_squared_error(actual, prediction)