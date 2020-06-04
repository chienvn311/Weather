import numpy as np
import pandas as pds
from sklearn.model_selection import train_test_split

# dir = "/Users/chienvn/PycharmProjects/Weather/Files/weather_exclude.csv"
# dir = "/Users/chienvn/PycharmProjects/Weather/Files/weather_include.csv"

def get_input(dir):
    weather_data = pds.read_csv(dir, header=0)
    weather_data = pds.get_dummies(weather_data.drop('Date', axis=1))
    weather_data = weather_data.drop('Rain_today_No', axis=1)
    weather_data = weather_data.drop('Rain_Tomorrow_No', axis=1)
    weather_data = weather_data.drop('Rainfall_Tomorrow', axis=1)
    dependence = np.array(weather_data['Rain_Tomorrow_Yes'])
    weather_data = weather_data.drop('Rain_Tomorrow_Yes', axis=1)
    independence = np.array(weather_data)
    train, test, train_lb, test_lb = train_test_split(independence, dependence, test_size=0.25, random_state=42)
    return train, test, train_lb, test_lb


# dir = "/Users/chienvn/PycharmProjects/Weather/Model/Classification/Input/new_weather_Ex.csv"
def get_input_new(dir):
    weather_data = pds.read_csv(dir, header=0)
    weather_data.fillna(weather_data.mean(), inplace=True)
    weather_data = weather_data.drop('Date', axis=1)
    dependence = np.array(weather_data['Rain_Tomorrow'])
    weather_data = weather_data.drop('Rain_Tomorrow', axis=1)
    weather_data = weather_data.drop('Time_of_maximum_wind_gust', axis=1)
    independence = np.array(weather_data)
    train, test, train_lb, test_lb = train_test_split(independence, dependence, test_size=0.25, random_state=42)
    return train, test, train_lb, test_lb


# dir = "/Users/chienvn/PycharmProjects/Weather/Model/Classification/Input/new_weather_Test.csv"
def get_input_test(dir):
    weather_data = pds.read_csv(dir, header=0)
    weather_data.fillna(weather_data.mean(), inplace=True)
    weather_data = weather_data.drop('Date', axis=1)
    test_lb = np.array(weather_data['Rain_Tomorrow'])
    weather_data = weather_data.drop('Rain_Tomorrow', axis=1)
    weather_data = weather_data.drop('Time_of_maximum_wind_gust', axis=1)
    test = np.array(weather_data)
    return test, test_lb

