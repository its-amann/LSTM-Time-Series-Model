---

# **Time Series Forecasting with LSTM Model (Multiple Series)**

This project demonstrates time-series forecasting using an LSTM (Long Short-Term Memory) model implemented with the Darts library for multiple time-series datasets.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Setup and Installation](#setup-and-installation)
3. [Data Preparation](#data-preparation)
4. [Model Building](#model-building)
5. [Evaluation](#evaluation)
6. [Results and Insights](#results-and-insights)
7. [Conclusion](#conclusion)
8. [References](#references)
9. [Author and Acknowledgments](#author-and-acknowledgments)

---

## **1. Introduction**
This project explores multi-series time-series forecasting using LSTM, leveraging key features in the Darts library. It supports multi-step forecasting and multiple time-series inputs for robust future predictions.

---

## **2. Setup and Installation**

### **Turn on GPU (if using Colab)**
Make sure the GPU is enabled in Colab for faster training.

### **Mount Google Drive (if using Colab)**
```python
from google.colab import drive
drive.mount('/content/drive')
```

### **Navigate to Working Directory**
```python
%cd /content/drive/MyDrive/Python - Time Series Forecasting/Deep Learning for Time Series Forecasting/LSTM
```

### **Install Required Libraries**
```python
!pip install darts
```

---

## **3. Data Preparation**

### **Import Required Libraries**
```python
# Standard Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error)

# Darts Functions
from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
```

---

### **Load Data**
```python
# Load the dataset
data = pd.read_csv("your_dataset.csv")
```

### **Data Preprocessing**
```python
# Preprocess the data (example)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
```

---

## **4. Model Building**

### **Model Initialization**
```python
model = RNNModel(
    model="LSTM",
    input_chunk_length=30,
    output_chunk_length=7,
    n_epochs=100,
    batch_size=16,
    hidden_size=32,
    dropout=0.2,
    model_name="LSTM_forecast",
    random_state=42,
)
```

### **Model Training**
```python
train, val = TimeSeries.split(data, 0.8)
model.fit(train)
```

---

## **5. Evaluation**

### **Model Predictions**
```python
forecast = model.predict(n=30, series=train)
```

### **Performance Metrics**
```python
mae = mean_absolute_error(val, forecast)
mse = mean_squared_error(val, forecast)
mape = mean_absolute_percentage_error(val, forecast)

print(f"MAE: {mae}, MSE: {mse}, MAPE: {mape}%")
```

---

## **6. Results and Insights**
- Visualization of Forecast Results
- Evaluation Graphs for Model Performance
- Insights from the Forecast

---

## **7. Conclusion**
- Summary of the Process
- Lessons Learned and Future Work Suggestions

---

## **8. References**
- [Darts Documentation](https://github.com/unit8co/darts)
- [LSTM Model Paper](https://arxiv.org/abs/1409.2329)

---

## **9. Author and Acknowledgments**
- Project Author: [Your Name]
- Acknowledgments: Contributors, Tutorials, and Online Resources

---
