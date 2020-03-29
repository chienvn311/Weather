import seaborn as sb
from matplotlib import pyplot as plt


def correlation(weather_data):
    c_mat = weather_data.corr()
    fig = plt.figure(figsize=(15, 15))
    sb.heatmap(c_mat, vmax=1, square=True)
    plt.show()


def histogram(weather_data):
    weather_data.hist(figsize=(12, 10))
    plt.show()