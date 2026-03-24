import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

print("Step 1: Downloading Data...")

# ==============================
# STEP 1: DATA COLLECTION
# ==============================
companies = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
data_dict = {}

for company in companies:
    data = yf.download(company, period="6mo")
    data_dict[company] = data['Close']

print("Data Downloaded!\n")

# ==============================
# STEP 2: TIME SERIES PLOT (SAVE)
# ==============================
for company in companies:
    plt.plot(data_dict[company], label=company)

plt.title("Stock Prices (Time Series)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

plt.savefig("time_series.png")  
plt.show()

# ==============================
# STEP 3: NORMALIZATION
# ==============================
scaler = MinMaxScaler()
scaled_data = {}

for company in companies:
    values = data_dict[company].values.reshape(-1,1)
    scaled = scaler.fit_transform(values)
    scaled_data[company] = scaled.flatten()

# ==============================
# STEP 4: STFT + SPECTROGRAM
# ==============================
spectrograms = {}

for company in companies:
    f, t, Zxx = stft(scaled_data[company], nperseg=32)
    spectrograms[company] = np.abs(Zxx)

# ==============================
# STEP 5: SPECTROGRAM PLOT (SAVE)
# ==============================
for company in companies:
    plt.figure()
    plt.pcolormesh(spectrograms[company])
    plt.title(f"Spectrogram - {company}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()

    plt.savefig(f"spectrogram_{company}.png")   
    plt.show()

# ==============================
# STEP 6: PREPARE DATA FOR CNN
# ==============================
X = []
y = []

data = scaled_data["RELIANCE.NS"]

window_size = 32

for i in range(len(data) - window_size - 1):
    segment = data[i:i+window_size]
    
    f, t, Zxx = stft(segment, nperseg=16)
    spec = np.abs(Zxx)
    
    X.append(spec)
    y.append(data[i+window_size])

X = np.array(X)
y = np.array(y)

# reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

print("Data ready for CNN:", X.shape)

# ==============================
# STEP 7: CNN MODEL
# ==============================
model = Sequential([
    Input(shape=X.shape[1:]),
    Conv2D(16, (2,2), activation='relu'),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("Training model...")

# ==============================
# STEP 8: TRAINING
# ==============================
model.fit(X, y, epochs=10, batch_size=32)

print("Training complete!\n")

# ==============================
# STEP 9: PREDICTION
# ==============================
pred = model.predict(X)

# Convert back to original scale
predicted = scaler.inverse_transform(pred)
actual = scaler.inverse_transform(y.reshape(-1,1))

# ==============================
# STEP 10: RESULT COMPARISON (SAVE)
# ==============================
plt.plot(actual, label="Actual")
plt.plot(predicted, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Prices")
plt.xlabel("Time")
plt.ylabel("Price")

plt.savefig("prediction.png")   
plt.show()

# ==============================
# STEP 11: MSE
# ==============================
mse = mean_squared_error(actual, predicted)
print("MSE:", mse)