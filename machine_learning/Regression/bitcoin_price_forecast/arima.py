from utils import *
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
import matplotlib.pyplot as plt
from pandas import read_html
from statsmodels.tsa.arima_model import ARIMA


#df = read_csv('BTC-USD.csv', usecols=['Date', 'Close'], index_col=0)

#Bitcoin 2017 ->
#df = read_html('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20170101&end=20200523')[2][::-1]
#Bitecoin All time
#df = read_html('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20200523')[2][::-1]
#Litecoin 2017 ->
#df = read_html('https://coinmarketcap.com/currencies/litecoin/historical-data/?start=20170101&end=20200523')[2][::-1]
#Litecoin All time




# df.drop(columns=['Open*', 'Low', 'High', 'Volume', 'Market Cap'],  axis=1, inplace=True)
# df.rename(columns={'Close**': 'Close'}, inplace=True)
# df['Date'] = df.apply(parse_date, axis=1)
# df = df.set_index('Date')
# df.index = pd.to_datetime(df.index)


#df = read_csv('Bitcoin-2017.csv', usecols=['Date', 'Close'], index_col=0)
#order = (1,1,3)
df = read_csv('Litecoin-2017.csv', usecols=['Date', 'Close'], index_col=0)
order = (1,1,2)

df.index = pd.to_datetime(df.index)
train_end = '2019-12-31'
test_start = '2020-01-01'
train = df.loc[:train_end]
test = df.loc[test_start:]
n_test = len(test)



predictions = list()
history = [x for x in train['Close']]



for i in range(n_test):
    model = ARIMA(history, order=order)
    model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0][0]
    predictions.append(yhat)
    new_obs = test.iloc[i].values[0]
    history.append(new_obs)

df_forecast = DataFrame(predictions, index=test.index, columns=['Close'])

print("rmse = ", rmse(test, df_forecast))
print("mae = ", mae(test, df_forecast))
plot_results(train, test, df_forecast)




# plt.plot(train, "Precio histórico")
# plt.plot(pd.concat([train.tail(1), test], axis=0), label="Precio real")
# plt.plot(pd.concat([train.tail(1), df_forecast], axis=0), label="Pronósticos")
# plt.legend()
# plt.show()










#Done
