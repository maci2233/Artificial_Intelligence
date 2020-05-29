from utils import *
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
from tensorflow.keras.models import model_from_json
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

crypto = "bi"

if crypto == "bi":
    df = read_csv('Bitcoin-2017.csv', usecols=['Date', 'Close'], index_col=0)
    order = (1,1,3)
    json_file = "lstm_residuals_bitcoin_model.json"
    weights_file = "lstm_residuals_bitcoin_model_weights.h5"
elif crypto == "li":
    df = read_csv('Litecoin-2017.csv', usecols=['Date', 'Close'], index_col=0)
    order = (1,1,2)
    json_file = "lstm_residuals_litecoin_model.json"
    weights_file = "lstm_residuals_litecoin_model_weights.h5"


df.index = pd.to_datetime(df.index)
train_end = '2019-12-31'
test_start = '2020-01-01'
train = df.loc[:train_end]
test = df.loc[test_start:]

history = [x for x in train['Close']]

model_arima = ARIMA(history, order=order)
model_fit = model_arima.fit(disp=0)
# plt.plot(history, color='red', label='real')
# plt.plot(model_fit.fittedvalues, color='blue', label='predicted')
# print(model_fit.resid[-1])
# print(history[-1] - model_fit.fittedvalues[-1])
# plt.legend()
# plt.show()
residuals = model_fit.resid

window_width = 4

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

load_model = True
if load_model:
    with open(json_file, "r") as f:
        json_model = f.read()
    model = model_from_json(json_model)
    model.load_weights(weights_file)
else:
    model = Sequential()
    model.add(LSTM(40,  input_shape=(x_train.shape[1], 1)))
    # model.add(Dropout(0.3))
    # model.add(LSTM(4, return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(LSTM(4))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=5, batch_size=32)

save_model = False
if save_model and not load_model:
    with open(json_file, "w") as f:
        f.write(model.to_json())
    model.save_weights(weights_file)



predictions = list()

x_test = np.array([y_train[-window_width:]]) #Getting the last N residuals for the first forecast

for i in range(len(test)):
    yhat_arima = model_fit.forecast()[0][0] #Arima forecast
    x_test_rs = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) #Reshaping to expected lstm input_shape
    resid_forecast = model.predict(x_test_rs) #predict the next residuals
    yhat_lstm = scaler.inverse_transform(resid_forecast)[0][0] #Inverse scale to get the final lstm forecast
    prediction = yhat_arima + yhat_lstm #add the forecasts to get final forecast
    predictions.append(prediction)
    y_true = test.iloc[i].values[0] #get the real price for that day
    residual = scaler.transform(np.array([y_true - yhat_arima]).reshape(-1, 1))[0][0] #Get the residual of the day
    x_test = np.append(x_test[0][1:], np.array([residual])) #Take out the oldest residual and add the new residual to the inputs of the lstm
    x_test = np.array([x_test])
    history.append(y_true) #Add the real price to the historical values
    model_arima = ARIMA(history, order=order) #Create the new ARIMA model using the history + the last real price
    model_fit = model_arima.fit(disp=0) #Retrain ARIMA

df_forecast = DataFrame(predictions, index=test.index, columns=['Close'])
#df_arimafitted = DataFrame(model_fit.fittedvalues, index=train.index, columns=['Close'])

#print(rmse(test_orig, df_forecast))
print("rmse = ", rmse(test, df_forecast))
print("mae = ", mae(test, df_forecast))
plot_results(train, test, df_forecast)

# plt.plot(train, label="True Train")
# plt.plot(pd.concat([train.tail(1), test], axis=0), label="True Test")
# plt.plot(pd.concat([train.tail(1), df_forecast], axis=0), label="Arima-LSTM Test")
# plt.show()












#Done
