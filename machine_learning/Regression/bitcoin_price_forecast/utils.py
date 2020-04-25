from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(test, predicted):
    return sqrt(mean_squared_error(test, predicted))
