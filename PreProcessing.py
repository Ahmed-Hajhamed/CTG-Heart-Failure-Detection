from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt
import numpy as np

def handle_missing_values(signal):
    # Replace NaNs with the mean of the signal
    return np.where(np.isnan(signal), np.nanmean(signal), signal)




def low_pass_filter(signal, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)



def clip_outliers(signal, lower_limit, upper_limit):
    return np.clip(signal, lower_limit, upper_limit)

# Define physiological ranges (adjust based on domain knowledge)
# fhr_signal_filtered = clip_outliers(fhr_signal_filtered, lower_limit=60, upper_limit=200)  # Example: FHR between 60-200 bpm
# uc_signal_filtered = clip_outliers(uc_signal_filtered, lower_limit=0, upper_limit=100)    # Example: UC between 0-100


def normalize_signal(signal):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(signal.reshape(-1, 1)).flatten()



def preprocess_signals(fhr_signal, uc_signal, sampling_rate):
    fhr_signal = handle_missing_values(fhr_signal)
    uc_signal = handle_missing_values(uc_signal)

    fhr_signal = low_pass_filter(fhr_signal, cutoff=0.5, fs=sampling_rate)
    uc_signal = low_pass_filter(uc_signal, cutoff=0.5, fs=sampling_rate)

    fhr_signal = clip_outliers(fhr_signal, lower_limit=60, upper_limit=200)
    uc_signal = clip_outliers(uc_signal, lower_limit=0, upper_limit=100)

    fhr_signal = normalize_signal(fhr_signal)
    uc_signal = normalize_signal(uc_signal)

    return fhr_signal, uc_signal