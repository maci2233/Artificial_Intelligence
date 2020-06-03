from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd


def rmse(test, predicted):
    return sqrt(mean_squared_error(test, predicted))


def mae(test, predicted):
    return mean_absolute_error(test, predicted)


def parse_date(row):
    months = {
        'Jan': '01',
        'Feb': '02',
        'Mar': '03',
        'Apr': '04',
        'May': '05',
        'Jun': '06',
        'Jul': '07',
        'Aug': '08',
        'Sep': '09',
        'Oct': '10',
        'Nov': '11',
        'Dec': '12',

    }
    date = row['Date'].split()
    return f"{date[-1]}-{months[date[0]]}-{date[1][:-1]}"


def plot_results(train, test, forecast):
    plt.plot(train, label="Precio histórico")
    plt.plot(pd.concat([train.tail(1), test], axis=0), label="Precio real")
    plt.plot(pd.concat([train.tail(1), forecast], axis=0), label="Pronósticos")
    plt.xlabel("Fecha")
    plt.ylabel("Precio de cierre USD")
    plt.legend()
    plt.show()
