from utils import *
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import model_from_json
from itertools import chain
import matplotlib.pyplot as plt


#df = read_csv('BTC-USD.csv', usecols=['Date', 'Close'], index_col=0)


#Bitcoin 2017 ->
#df = read_html('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20170101&end=20200523')[2][::-1]
#Bitecoin All time
#df = read_html('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20200523')[2][::-1]
#Litecoin 2017 ->
#df = read_html('https://coinmarketcap.com/currencies/litecoin/historical-data/?start=20170101&end=20200523')[2][::-1]

# df.drop(columns=['Open*', 'Low', 'High', 'Volume', 'Market Cap'],  axis=1, inplace=True)
# df.rename(columns={'Close**': 'Close'}, inplace=True)
# df['Date'] = df.apply(parse_date, axis=1)
# df = df.set_index('Date')
# df.index = pd.to_datetime(df.index)

#df = read_csv('Bitcoin-2017.csv', usecols=['Date', 'Close'], index_col=0)
df = read_csv('Litecoin-2017.csv', usecols=['Date', 'Close'], index_col=0)

df.index = pd.to_datetime(df.index)
train_end = '2019-12-31'
test_start = '2020-01-01'

train_orig = df.loc[:train_end]
test_orig = df.loc[test_start:]

scaler = MinMaxScaler()
df = DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

window_width = 4

df = pd.concat([df.shift(i) for i in range(window_width, 0 , -1)] + [df], axis=1)[window_width:]
df.columns = [column + '-' + str(t) for column, t in zip(df.columns, range(window_width, 0, -1))] + ['Close']

train = df.loc[:train_end]
test = df.loc[test_start:]

#train = DataFrame(scaler.fit_transform(train), index=train.index, columns=train.columns)

x_train = train.drop('Close', axis=1).to_numpy()
y_train = train.pop('Close').to_numpy().reshape(-1, 1)

#x_train = scaler.fit_transform(x_train)
#y_train = scaler.fit_transform(y_train.reshape(-1, 1))

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


load_model = True
if load_model:
    with open("lstm_model.json", "r") as f:
        json_model = f.read()
    model = model_from_json(json_model)
    model.load_weights("lstm_model_weights.h5")

else:
    model = Sequential()
    model.add(LSTM(40, input_shape=(x_train.shape[1], 1)))
    #model.add(Dropout(0.3))
    #model.add(LSTM(4, return_sequences=True))
    #model.add(Dropout(0.3))
    #model.add(LSTM(4))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=5, batch_size=32)


save_model = False
if save_model and not load_model:
    with open("lstm_model.json", "w") as f:
        f.write(model.to_json())
    model.save_weights("lstm_model_weights.h5")


x_test = test.drop('Close', axis=1).to_numpy()
y_test = test.pop('Close').to_numpy().reshape(-1, 1)


x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
forecast = model.predict(x_test)

forecast = np.array(list(chain.from_iterable(forecast)))
forecast = scaler.inverse_transform(forecast.reshape(-1, 1))

df_forecast = DataFrame(forecast, index=test_orig.index, columns=['Close'])



print("rmse = ", rmse(test_orig, df_forecast))
print("mae = ", mae(test_orig, df_forecast))

plot_results(train_orig, test_orig, df_forecast)

# plt.plot(train_orig)
# plt.plot(pd.concat([train_orig.tail(1), test_orig], axis=0))
# plt.plot(pd.concat([train_orig.tail(1), df_forecast], axis=0))
# plt.show()









#Done
