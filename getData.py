import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def get_data(data_type):
    switcher = {
        0: "/Users/chienvn/PycharmProjects/Weather/Files/weather_include.csv",
        1: "/Users/chienvn/PycharmProjects/Weather/Files/weather_exclude.csv",
        2: "/Users/chienvn/PycharmProjects/Weather/Files/weather_consider.csv",
    }
    location = switcher.get(data_type, "/Users/chienvn/PycharmProjects/Weather/weather_include.csv")
    print(location)
    weather_data = pd.read_csv(location, error_bad_lines=False)
    return weather_data


def slip_data(weather_data, col):
    dependence = np.array(weather_data[col])
    weather_data = weather_data.drop(col, axis=1)
    if col == "Rain_Tomorrow_Yes":
        weather_data = weather_data.drop("Rainfall_Tomorrow", axis=1)
        print('skip')
    elif col == "Rainfall_Tomorrow":
        weather_data = weather_data.drop("Rain_Tomorrow_Yes", axis=1)
    independence = np.array(weather_data)
    train, test, train_lb, test_lb = train_test_split(independence, dependence, test_size=0.25, random_state=42)
    return train, test, train_lb, test_lb


def get_train_lb(col):
    switcher = {
        0: "Rain_Tomorrow_Yes",
        1: "Rain_today_Yes",
        2: "Rainfall",
        3: "Rainfall_Tomorrow",
    }
    print(switcher.get(col, "Rain_Tomorrow_Yes"))
    return switcher.get(col, "Rain_Tomorrow_Yes")


def one_hot_encode(df):
    return pd.get_dummies(df)


def export_pandas_csv(weather_data):
    cwd = os.getcwd()
    direction = cwd + "/report.csv"
    weather_data.to_csv(direction, header=True, index=False)


def export_numpy_csv(weather_data):
    np.savetxt("report.csv", weather_data, delimiter=",")


def clear_data(data):
    new_data = one_hot_encode(data.drop('Date', axis=1))
    new_data = new_data.drop('Rain_today_No', axis=1)
    new_data = new_data.drop('Rain_Tomorrow_No', axis=1)
    return new_data

