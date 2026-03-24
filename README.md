📈 Stock Price Prediction using STFT and CNN
👩‍💻 Student Details
Name: Cilla Elsa Binoy 
Register Number: TCR24CS021

📌 Project Description
This project demonstrates how financial time series data can be analyzed using signal processing and deep learning techniques.

Stock price data is transformed into a time–frequency representation (spectrogram) using the Short-Time Fourier Transform (STFT). A Convolutional Neural Network (CNN) is then trained on this representation to predict future stock prices.

🎯 Objective
Treat stock data as a signal
Extract hidden patterns using STFT
Use CNN for prediction
Compare predicted vs actual prices
📥 Inputs
Stock price data of:

RELIANCE.NS
TCS.NS
INFY.NS
Data collected using Yahoo Finance

📤 Outputs
📊 Time Series Plot
Shows variation of stock prices over time.

Time Series

🌈 Spectrogram
Represents time-frequency characteristics of stock data.

Spectrogram

📈 Prediction vs Actual
Comparison between predicted and real stock prices.

Prediction

⚙️ Technologies Used
Python
NumPy
Pandas
Matplotlib
SciPy (STFT)
TensorFlow / Keras
yFinance API
🧪 How to Run
Step 1: Install dependencies
pip install numpy pandas matplotlib scipy scikit-learn tensorflow yfinance
Step 2: Run the program
py -3.11 stock_project.py
📊 Methodology
Collect stock data
Normalize data
Apply STFT to generate spectrogram
Prepare data for CNN
Train CNN model
Predict future stock prices
Evaluate using MSE
📈 Results
Spectrogram reveals hidden patterns in stock data
CNN model learns trends effectively
Predicted values closely follow actual prices
🧠 Conclusion
This project shows that combining signal processing and deep learning provides an effective approach for financial time series forecasting.

📚 References
Yahoo Finance
IEEE Research Papers on Stock Prediction
Deep Learning for Time Series Forecasting
🚀 Future Improvements
Use multiple features (volume, index, exchange rate)
Improve CNN architecture
Try LSTM models for better accuracy
