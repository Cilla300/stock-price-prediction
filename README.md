# 📈 Pattern Recognition for Financial Time Series Forecasting

👩‍💻 Student Details
Name: Cilla Elsa Binoy Register Number: TCR24CS021

## 🧠 Project Overview

This project explores how **time–frequency signal processing** and **deep learning (CNN)** can be used to predict stock prices from financial time series data.

Financial data such as stock prices are treated as **signals**, transformed into a **spectrogram (image)** using Short-Time Fourier Transform (STFT), and then fed into a **Convolutional Neural Network (CNN)** for prediction.

---

## 🎯 Objective

* Convert financial time series data into a **time–frequency representation**
* Extract hidden patterns using **spectrograms**
* Train a **CNN model** to predict future stock prices
* Evaluate model performance using **Mean Squared Error (MSE)**

---

## 📊 Dataset

Stock data is collected using:

* Yahoo Finance API (via `yfinance` library)

### Companies Used:

* RELIANCE.NS
* TCS.NS
* INFY.NS

### Features:

* Closing Price (primary signal used for modeling)

---

## ⚙️ Methodology

### 1. Data Collection

* Download historical stock data using `yfinance`

### 2. Preprocessing

* Extract closing prices
* Normalize data using **MinMaxScaler**

### 3. Signal Processing

#### 🔹 Fourier Transform

* Convert time-domain signal to frequency domain

#### 🔹 Short-Time Fourier Transform (STFT)

* Apply sliding window to analyze non-stationary signals

#### 🔹 Spectrogram

* Generate time–frequency representation:

  * X-axis → Time
  * Y-axis → Frequency
  * Color → Energy

---

## 🧩 Model Architecture (CNN)

* Input: Spectrogram images
* Layers:

  * Conv2D
  * MaxPooling2D
  * Flatten
  * Dense (Fully Connected)
* Output: Predicted stock price

---

## 🔄 Workflow Pipeline

Time Series Data
→ Normalization
→ STFT
→ Spectrogram
→ CNN Model
→ Prediction

---

## 🧪 Training & Testing

* Data split into:

  * 80% Training
  * 20% Testing
* Model trained using:

  * Loss Function: Mean Squared Error (MSE)
  * Optimizer: Adam

---

## 📈 Results

### Outputs Generated:

* Time Series Plot
* Frequency Spectrum
* Spectrogram Images
* Actual vs Predicted Graph

### Evaluation Metric:

* Mean Squared Error (MSE)

---

## 📌 Key Insights

* Financial data behaves like a **non-stationary signal**
* Spectrogram reveals hidden patterns not visible in raw data
* CNN can learn useful representations from spectrogram images

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* SciPy
* Scikit-learn
* TensorFlow / Keras
* yFinance

---

## 🚀 How to Run

1. Install required libraries:

```
pip install numpy pandas matplotlib yfinance scipy scikit-learn tensorflow
```

2. Run the script:

```
python project.py
```

3. Output:

* Graphs will be displayed and saved as images
* Model will print MSE

---

## ⚠️ Notes

* Only closing price was used for simplicity
* Model performance can improve with more features and data
* Spectrogram visualization improved using logarithmic scaling

---

## 📚 References

1. Y. Zhang & C. Aggarwal – Stock Market Prediction Using Deep Learning
2. A. Tsantekidis – Deep Learning for Financial Time Series
3. Hochreiter & Schmidhuber – LSTM (1997)
4. Borovykh et al. – CNN for Time Series Forecasting

---

## 📌 Conclusion

This project demonstrates that:

* Financial time series can be analyzed as signals
* Time–frequency transformation is powerful
* CNN models can effectively predict stock price trends

---

