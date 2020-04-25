from utils import rmse
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = read_csv('BTC-USD.csv', usecols=['Date', 'Close'], index_col=0)
df.index = pd.to_datetime(df.index)

train_end = '2019-12-31'
test_start = '2020-01-01'
train = df.loc[:train_end]
test = df.loc[test_start:]
history = [x for x in train['Close']]

model = ARIMA(history, order=(1,0,3))
model_fit = model.fit(disp=0)
# plt.plot(history, color='red', label='real')
# plt.plot(model_fit.fittedvalues, color='blue', label='predicted')
# print(model_fit.resid[-1])
# print(history[-1] - model_fit.fittedvalues[-1])
# plt.legend()
# plt.show()
residuals = model_fit.resid[1:]

window_width=4

#train2 is the dataframe that contains all the residuals
train2 = DataFrame(residuals, index=train.drop(train.index[0]).index, columns=['Close'])
scaler = MinMaxScaler().fit(train2)
train2 = DataFrame(scaler.transform(train2), index=train2.index, columns=train2.columns)

train2 = pd.concat([train2.shift(i) for i in range(window_width, 0 , -1)] + [train2], axis=1)[window_width:]
train2.columns = [column + '-' + str(t) for column, t in zip(train2.columns, range(window_width, 0, -1))] + ['Close']

x_train = train2.drop('Close', axis=1).to_numpy()
y_train = train2.pop('Close').to_numpy()
# x_train = scaler.fit_transform(x_train)
# y_train = scaler.fit_transform(y_train.reshape(-1, 1))

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


model = Sequential()
model.add(LSTM(4, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(LSTM(4, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(4))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=5, batch_size=32)

predictions = []

x_test = np.array([y_train[-window_width:]])

for i in range(1):#len(test)):
    yhat_arima = model_fit.forecast()[0][0]
    x_test_rs = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    forecast = model.predict(x_test_rs)
    yhat_lstm = scaler.inverse_transform(forecast)[0][0]
    predictions.append(yhat_arima + yhat_lstm)
    #falta calcular el residual y usarlo para la siguiente prediccion de lstm (x_test)

























print("Done")
