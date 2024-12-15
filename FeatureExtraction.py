import numpy as np
from scipy.signal import find_peaks
import ExtractData
# Example FHR and UC signals
# fhr_signal = np.array([130, 135, 125, 110, 115, 125, 130])
# uc_signal = np.array([5, 7, 20, 50, 15, 7, 5, 15, 5])
fhr_signal, uc_signal = ExtractData.extract_data()
# Detect UC peaks
uc_peaks, properties = find_peaks(uc_signal, height=10)  # Peaks >10

# Define baseline FHR (mean of the signal before UC)
baseline_fhr = np.mean(fhr_signal[:uc_peaks[0]])

# Extract features for each UC
features = []
for peak in uc_peaks:
    onset = max(0, peak - 2)  # Example: 2 time points before peak
    offset = min(len(fhr_signal), peak + 2)  # Example: 2 time points after peak
    
    # FHR response to UC
    during_uc = fhr_signal[onset:offset]
    min_fhr = np.min(during_uc)  # Lowest FHR during UC
    deceleration_depth = baseline_fhr - min_fhr  # Drop in FHR
    recovery_time = np.argmax(fhr_signal[offset:] >= baseline_fhr) if offset < len(fhr_signal) else 0
    
    # Time lag between UC peak and FHR nadir
    fhr_nadir_index = np.argmin(during_uc)
    time_lag = (fhr_nadir_index + onset) - peak
    
    # Store features for this UC
    features.append({
        "Baseline FHR": baseline_fhr,
        "Min FHR": min_fhr,
        "Deceleration Depth": deceleration_depth,
        "Recovery Time": recovery_time,
        "Time Lag (Peak to Nadir)": time_lag
    })

# Convert to structured data (e.g., pandas DataFrame)
import pandas as pd
features_df = pd.DataFrame(features)
print(features_df)

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plt.plot(fhr_signal, label='FHR (bpm)', color='blue')
plt.plot(uc_signal, label='UC (arbitrary units)', color='orange')
plt.axhline(y=baseline_fhr, color='green', linestyle='--', label='Baseline FHR')
plt.scatter(uc_peaks, uc_signal[uc_peaks], color='red', label='UC Peaks')
plt.legend()
plt.show()
