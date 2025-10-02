# LSTM-Stock-Price-Prediction, Time Series Prediction with Deep Learning

This project explores different deep learning models for predicting the next day's stock/financial time series values for AAPL stock. The models implemented include **Baseline LSTM**, **Stacked LSTM**, **GRU**, and **1D CNN**. Each model is trained on a sliding window of past observations to forecast the next value.

---

## Models Implemented

### 1. Baseline LSTM

```python
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(window_size, features)))
model.add(Dropout(0.2))
model.add(Dense(1))  # predicting 1 value (next day's close)

model.compile(optimizer='adam', loss='mse')
model.summary()
```

**Evaluation Metrics:**

* RMSE: 0.0150
* MAE: 0.0117
* MAPE: 1.79%

---

### 2. Stacked LSTM

```python
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, features)))
model.add(LSTM(units=50, return_sequences=False))  # Last LSTM layer
model.add(Dense(1))  # Predict next day's close

model.compile(optimizer='adam', loss='mse')
model.summary()
```

**Evaluation Metrics:**

* RMSE: 0.0120
* MAE: 0.0092
* MAPE: 1.40%

---

### 3. GRU

```python
from keras.layers import GRU

model = Sequential()
model.add(GRU(units=50, return_sequences=False, input_shape=(window_size, features)))
model.add(Dropout(0.2))
model.add(Dense(1))  # predicting 1 value (next day's close)

model.compile(optimizer='adam', loss='mse')
model.summary()
```

**Evaluation Metrics:**

* RMSE: 0.0125
* MAE: 0.0097
* MAPE: 1.47%

---

### 4. 1D CNN

```python
from keras.layers import Conv1D, MaxPooling1D, Flatten

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1))  # predicting next day's close

model.compile(optimizer='adam', loss='mse')
model.summary()
```

**Evaluation Metrics:**

* RMSE: 0.0350
* MAE: 0.0291
* MAPE: 4.50%

---

## Model Performance Comparison

| Model         | RMSE   | MAE    | MAPE  |
| ------------- | ------ | ------ | ----- |
| Baseline LSTM | 0.0150 | 0.0117 | 1.79% |
| Stacked LSTM  | 0.0120 | 0.0092 | 1.40% |
| GRU           | 0.0125 | 0.0097 | 1.47% |
| 1D CNN        | 0.0350 | 0.0291 | 4.50% |

**Observations:**

* **Stacked LSTM** achieved the best performance, capturing complex temporal patterns effectively.
* **GRU** performed closely to Stacked LSTM, showing efficiency in sequential modeling.
* **1D CNN** had the highest error, indicating convolutional approaches may be less suitable for this task.

---
