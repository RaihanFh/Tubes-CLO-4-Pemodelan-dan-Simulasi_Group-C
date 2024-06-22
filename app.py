import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

st.title("Prediksi Harga Saham dengan Kombinasi Machine Learning dan Gerakan Brown Geometrik (GBM)")

st.write("""
### Anggota Kelompok:
- Adhie Haqqi Ramadhani S. - 1301213312
- Naufal Alfarisi - 1301213452
- Mufidah Alfiah - 1301184180
- Raihan Fadhilah Hafiizh - 1301213113
""")

# Input fields for user parameters
st.header("Input Parameters")
ticker = st.text_input("Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

# Button to run the prediction
if st.button("Run Prediction"):
    # Load and process the data
    st.write("Loading data...")
    # Placeholder for data loading function (you should replace this with actual data loading)
    data = pd.DataFrame()  # replace with actual data loading

    # Placeholder for your existing prediction code
    st.write("Running prediction...")
    # Example: Fit a linear regression model (replace with your model)
    X = data[['Open', 'High', 'Low', 'Close']].values
    y = data['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression().fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    # Display results
    st.write(f"RMSE: {rmse}")

    # Visualization
    st.header("Prediction vs Actual")
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    st.pyplot(plt)

    st.write("Done.")

