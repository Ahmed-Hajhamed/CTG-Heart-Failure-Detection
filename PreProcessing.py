from scipy.signal import butter, filtfilt, find_peaks, resample
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Load signals (FHR and UC)
fhr_signal = np.array([...])  # Replace with actual data
uc_signal = np.array([...])

# Step 1: Interpolate missing values
fhr_signal = pd.Series(fhr_signal).interpolate(method='linear').to_numpy()
uc_signal = pd.Series(uc_signal).interpolate(method='linear').to_numpy()

# Step 2: Normalize signals
scaler = MinMaxScaler() #use StandardScaler() for z score (zero mean and unit variance)
fhr_signal = scaler.fit_transform(fhr_signal.reshape(-1, 1)).flatten()
uc_signal = scaler.fit_transform(uc_signal.reshape(-1, 1)).flatten()

# Step 3: Filter signals (low-pass filter)
def low_pass_filter(signal, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

fhr_signal = low_pass_filter(fhr_signal, cutoff=0.3, fs=4)  # Example frequency: 4 Hz
uc_signal = low_pass_filter(uc_signal, cutoff=0.3, fs=4)

# Step 4: Resample and synchronize
new_length = len(fhr_signal)
uc_signal = resample(uc_signal, new_length)

# Step 5: Segment signals
def segment_signal(signal, window_size, overlap):
    step = int(window_size * (1 - overlap))
    return [signal[i:i+window_size] for i in range(0, len(signal) - window_size + 1, step)]

window_size = 60
overlap = 0.5
fhr_windows = segment_signal(fhr_signal, window_size, overlap)
uc_windows = segment_signal(uc_signal, window_size, overlap)
