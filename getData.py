import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt


def get_data():
    weather_data = pd.read_csv('/Users/chienvn/PycharmProjects/Weather/weather.csv', error_bad_lines=False)
    return weather_data


def slip_data(weather_data):
    dependence = np.array(weather_data['Rainfall'])
    weather_data = weather_data.drop('Rainfall', axis=1)
    independence = np.array(weather_data)
    train, test, train_lb, test_lb = train_test_split(independence, dependence, test_size=0.25, random_state=42)
    return train, test, train_lb, test_lb


def one_hot_encode(df):
    return pd.get_dummies(df)


def correlation():
    weather_data = get_data()
    c_mat = weather_data.corr()
    fig = plt.figure(figsize=(15, 15))
    sb.heatmap(c_mat, vmax=1, square=True)
    plt.show()


def histogram():
    weather_data = get_data()
    weather_data.hist(figsize=(12, 10))
    plt.show()