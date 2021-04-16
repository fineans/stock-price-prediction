from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime as dt
plt.style.use('fivethirtyeight')

ticker = input("Stock Symbol: ")
end = dt.datetime.now()
start = end  - dt.timedelta(days=16)
dp = pdr.get_data_yahoo(ticker, start, end)
dp.to_csv('Data.csv')
df = pd.read_csv('Data.csv')
df["Change"] = df.Close - df.Open

days = list()
adj_close_prices = list()

df_days = df.loc[:, 'Date']
df_adj_close = df.loc[:, 'Adj Close']

for day in df_days:
  days.append([int(day.split('-')[2])])

for adj_close_price in df_adj_close:
  adj_close_prices.append(float(adj_close_price))

lin_svr = SVR(kernel='linear', C=1000.0)
lin_svr.fit(days, adj_close_prices)

poly_svr = SVR(kernel='poly', C=1000.0, degree=2)
poly_svr.fit(days, adj_close_prices)

rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=0.85)
rbf_svr.fit(days, adj_close_prices)

day = [[24]] # ->enter next trading sessions day number
#print('The RBF SVR Predicted price: ', rbf_svr.predict(day))
#print('The POLY SVR Predicted price: ', poly_svr.predict(day))
#print('The LINEAR SVR Predicted price: ', lin_svr.predict(day))
avg = (rbf_svr.predict(day)+poly_svr.predict(day)+lin_svr.predict(day))/3
out = avg[0]

print('The Predicted price is Dollars', out)