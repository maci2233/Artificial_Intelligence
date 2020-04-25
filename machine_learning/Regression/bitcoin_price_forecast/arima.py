from utils import rmse
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA


df = read_csv('BTC-USD.csv', usecols=['Date', 'Close'], index_col=0)
df.index = pd.to_datetime(df.index)
train_end = '2019-12-31'
test_start = '2020-01-01'
train = df.loc[:train_end]
test = df.loc[test_start:]
n_test = len(test)

predictions = list()
history = [x for x in train['Close']]

for i in range(n_test):
    if i % 7 == 0: #REENTRENANDO CADA DIA con i % 1
        model = ARIMA(history, order=(1,0,3)) #FALTA CHECAR ESTE PEDO Y VER COMO JALA Y TODO ESO
        model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0][0]
    predictions.append(yhat)
    new_obs = test.iloc[i].values[0]
    history.append(new_obs)

df_forecast = DataFrame(predictions, index=test.index, columns=['Close'])

print(rmse(test, df_forecast))

plt.plot(train)
plt.plot(pd.concat([train.tail(1), test], axis=0))
plt.plot(pd.concat([train.tail(1), df_forecast], axis=0))
plt.show()
