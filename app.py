import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import ta
import matplotlib.pyplot as plt

st.title("Prediksi Harga Saham AGRO.JK (Bank Rakyat Indonesia) dengan Kombinasi Machine Learning dan Gerakan Brown Geometrik (GBM)")
st.write("Anggota Kelompok:")
st.write("* Adhie Haqqi Ramadhani S. -  1301213312")
st.write("* Naufal Alfarisi - 1301213452")
st.write("* Mufidah Alfiah - 1301184180")
st.write("* Raihan Fadhilah Hafiizh - 1301213113")

# Input untuk memilih tanggal mulai dan tanggal akhir
start_date = st.date_input("Pilih tanggal mulai", value=pd.to_datetime("2021-01-01"))
end_date = st.date_input("Pilih tanggal akhir", value=pd.to_datetime("2023-12-31"))

@st.cache
def load_data(start, end):
    data = yf.download("AGRO.JK", start=start, end=end)
    data = data.interpolate(method='linear')
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)
    data['SMA'] = ta.trend.sma_indicator(close=data['Close'], window=14)
    data['EMA'] = ta.trend.ema_indicator(close=data['Close'], window=14)
    data['RSI'] = ta.momentum.rsi(close=data['Close'], window=14)
    data.dropna(inplace=True)
    return data

data = load_data(start_date, end_date)

st.subheader('Data Saham AGRO.JK')
st.write(data.head(20))

features = ['SMA', 'EMA', 'RSI']
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Log_Return'], label='Log Return')
plt.title('Daily Log Returns of AGRO.JK')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.grid(True)
st.pyplot(plt)

plt.figure(figsize=(10, 6))
plt.plot(data.index, data['SMA'], label='SMA')
plt.plot(data.index, data['EMA'], label='EMA')
plt.plot(data.index, data['RSI'], label='RSI')
plt.title('Technical Indicators of AGRO.JK')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid(True)
st.pyplot(plt)

def train_model(data, features):
    X = data[features]
    y = data['Log_Return']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2

model, mse, r2 = train_model(data, features)

st.subheader('Evaluasi Model')
st.write(f"Mean Squared Error: {mse}")
st.write(f"R-squared: {r2}")

def calculate_volatility(data):
    daily_returns = np.log(data["Adj Close"].pct_change() + 1)
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    return annualized_volatility

def gbm_sim(spot_price, volatility, steps, model, features, data):
    dt = 1 / 252
    paths = [spot_price]
    drift = model.predict(data[features])

    for i in range(len(drift)):
        random_shock = np.random.normal() * np.sqrt(dt)
        new_price = paths[-1] * np.exp((drift[i] - 0.5 * volatility**2) * dt + volatility * random_shock)
        paths.append(new_price)

    return paths[:-1], drift

steps = len(data)
spot_price = data["Adj Close"].iloc[0]
volatility = calculate_volatility(data)
simulated_paths, drifts = gbm_sim(spot_price, volatility, steps, model, features, data)

plt.figure(figsize=(10, 6))
index = data.index
plt.plot(index, simulated_paths[:len(index)], label='Predicted')
plt.plot(index, data['Adj Close'].values, label='Actual')
plt.xlabel("Time Step")
plt.ylabel("Stock Price")
plt.title("Simulated Stock Price Paths")
plt.legend()
st.pyplot(plt)

labels = ['Predicted Drift', 'Actual Drift', 'Absolute Error']
fig, ax = plt.subplots(1, 3, figsize=(10, 2))
ax[0].plot(drifts[:len(index)])
ax[1].plot(data['Log_Return'].values[:len(index)])
ax[2].plot([abs(i - j) for (i, j) in zip(drifts, data['Log_Return'].values[:len(index)])])
_ = [ax[i].set_title(j) for i, j in enumerate(labels)]
st.pyplot(fig)
