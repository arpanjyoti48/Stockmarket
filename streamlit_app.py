import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf




st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

google_data = yf.download(stock, start, end)

model = load_model("Stock_prediction_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# x_test = x_test.reset_index()
if 'Close' in x_test.columns:
    column_name = 'Close'
else:
    column_name = x_test.columns[0]  # Use the first column if 'Close' doesn't exist

print(f"Using column: {column_name}")
scaled_data = scaler.fit_transform(x_test[[column_name]])

print(x_test.head())
st.subheader(x_test.columns[0])
st.subheader(x_test.head())
print(x_test.columns)

# scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[[stock]])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)



future_input = scaled_data[-100:].reshape(1, 100, 1)  # Last 100 data points
future_predictions = []

for _ in range(90):  # Predict for the next 90 days
    pred = model.predict(future_input)
    future_predictions.append(pred[0][0])
    # Append the prediction and roll the input
    future_input = np.append(future_input[:, 1:, :], [[pred[0]]], axis=1)

# Inverse-transform future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(google_data.index[-1], periods=90 + 1, freq='B')[1:]  # Next 90 business days
future_plot = pd.DataFrame({'Future Predictions': future_predictions.flatten()}, index=future_dates)

# Display the future predictions as a line chart
st.subheader("Future Predictions for the Next 30 Days")
st.line_chart(future_plot)
st.write(future_plot)

# Extend the curve by combining historical and future predictions into one DataFrame
combined_data = pd.concat(
    [google_data['Close'], future_plot['Future Predictions']],
    axis=0
)
fig = plt.figure(figsize=(15, 6))
plt.plot(google_data.index, google_data['Close'], label="Historical Prices", color="blue")
plt.plot(future_plot.index, future_plot['Future Predictions'], label="Future Predictions", color="orange", linestyle="--")
plt.axvline(x=google_data.index[-1], color="gray", linestyle="--", label="Prediction Start")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"Extended Stock Price Prediction for {stock}")
st.pyplot(fig)
