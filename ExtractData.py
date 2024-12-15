import wfdb
# Visualize signals
import matplotlib.pyplot as plt

def extract_data():
    record_name = 'cardiotocography-dataset/1001'
    # Read the record
    record = wfdb.rdrecord(record_name)

    # Extract signals
    signals = record.p_signal  # Multi-dimensional array: each column is a signal
    sampling_rate = record.fs  # Sampling frequency
    signal_names = record.sig_name  # Signal names (e.g., FHR, UC)

    # Example of extracting FHR and UC signals
    fhr_signal = signals[:, signal_names.index('FHR')]
    uc_signal = signals[:, signal_names.index('UC')]
    return fhr_signal, uc_signal

fhr_signal, uc_signal = extract_data()
plt.figure(figsize=(10, 5))

# FHR Signal
plt.subplot(2, 1, 1)
plt.plot(fhr_signal, label='FHR', color='blue')
plt.title("Fetal Heart Rate (FHR)")
plt.ylabel("BPM")
plt.grid()
plt.legend()

# UC Signal
plt.subplot(2, 1, 2)
plt.plot(uc_signal, label='Uterine Contractions (UC)', color='green')
plt.title("Uterine Contractions (UC)")
plt.ylabel("Intensity")
plt.xlabel("Time (samples)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
