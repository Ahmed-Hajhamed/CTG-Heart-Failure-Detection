import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def calculate_features(FHR, UC, sampling_rate=1, long_term_window=600,):
    """
    Calculate the first 11 features based on FHR and UC time-series data.
    Args:
    - FHR (array-like): Fetal Heart Rate time-series data.
    - UC (array-like): Uterine Contractions time-series data.
    - sampling_rate (float): Sampling rate of the data (in Hz, i.e., data points per second).
    - long_term_window (int): The length of the long-term window (in seconds, default is 600 seconds = 10 minutes).
    - show_histogram (bool): Whether to display a histogram of FHR values.
    """
    def variability(signal, window_size):
        """Calculate variability over sliding windows."""
        return np.std([np.std(signal[i:i+window_size]) for i in range(0, len(signal)-window_size, window_size)])

    features = {}

    # 1. FHR Baseline (LB)
    baseline_window = 10 * sampling_rate * 60  # 10 minutes in samples
    features['LB'] = np.mean(FHR[-baseline_window:]) if len(FHR) > baseline_window else np.mean(FHR)

    # 2. Number of Accelerations per second (AC)
    acceleration_threshold = 15  # 15 bpm above baseline
    ac_peaks, _ = find_peaks(FHR, height=features['LB'] + acceleration_threshold)
    features['AC'] = len(ac_peaks) / len(FHR)

    # 3. Number of Fetal Movements per second (FM)
    movement_threshold = 10  # Smaller threshold for movements
    fm_peaks, _ = find_peaks(FHR, height=features['LB'] + movement_threshold)
    features['FM'] = len(fm_peaks) / len(FHR)

    # 4. Number of Uterine Contractions per second (UC)
    uc_peaks, _ = find_peaks(UC, height=0.5)
    features['UC'] = len(uc_peaks) / len(UC)

    # 5. Number of Light Decelerations per second (DL)
    light_decel = [1 for i in range(1, len(FHR)) if FHR[i] < FHR[i-1] - 15]
    features['DL'] = len(light_decel) / len(FHR)

    # 6. Number of Severe Decelerations per second (DS)
    severe_decel = [1 for i in range(1, len(FHR)) if FHR[i] < FHR[i-1] - 30]
    features['DS'] = len(severe_decel) / len(FHR)

    # 7. Number of Prolonged Decelerations per second (DP)
    prolonged_decel = [
        1 for i in range(1, len(FHR))
        if FHR[i] < FHR[i-1] - 30 and
        (i - np.where(FHR[:i] < FHR[i-1] - 30)[0][0]) > (2 * sampling_rate * 60)
    ]
    features['DP'] = len(prolonged_decel) / len(FHR)

    # 8. Percentage of Time with Abnormal Short-Term Variability (ASTV)
    abnormal_stv = np.mean([np.std(FHR[i:i+sampling_rate]) < 5 for i in range(0, len(FHR)-sampling_rate, sampling_rate)])
    features['ASTV'] = abnormal_stv

    # 9. Mean Short-Term Variability (MSTV)
    short_term_window = sampling_rate * 60  # 1-minute window in samples
    features['MSTV'] = variability(FHR, short_term_window)

    # 10. Percentage of Time with Abnormal Long-Term Variability (ALTV)
    long_term_samples = long_term_window * sampling_rate
    abnormal_ltv = np.mean([
        np.std(FHR[i:i+long_term_samples]) < 5 for i in range(0, len(FHR)-long_term_samples, long_term_samples)
    ])
    features['ALTV'] = abnormal_ltv

    # 11. Mean Long-Term Variability (MLTV)
    window_size = long_term_samples
    long_term_variability = [
        np.std(FHR[i:i+window_size]) for i in range(0, len(FHR) - window_size + 1, window_size)
    ]
    features['MLTV'] = np.mean(long_term_variability)

    return features

# Example usage
FHR = np.random.randn(3600) * 10 + 120  # Simulated FHR data (1 hour of data)
UC = np.random.randn(3600) * 0.5 + 1    # Simulated UC data (1 hour of data)


