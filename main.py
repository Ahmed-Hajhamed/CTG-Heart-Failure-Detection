from re import X
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGridLayout, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QLineEdit, QTableWidgetItem,
    QPushButton, QComboBox, QSlider, QFileDialog, QSpacerItem, QSizePolicy, QGraphicsScene, QCheckBox, QTabWidget
)
from sklearn.preprocessing import MinMaxScaler
from PreProcessing import handle_missing_values, low_pass_filter, normalize_signal, clip_outliers
from ui import ui
import wfdb
import numpy as np
from scipy.signal import find_peaks
# from imblearn.over_sampling import SMOTE
from qt_material import apply_stylesheet
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


scaler = MinMaxScaler()

class main(ui):
    def __init__(self):
        super().__init__()
        
        self.model, self.columns_name = model_creator()
        self.fhr_signal = None
        self.uc_signal = None
        self.features = None
        self.fatel_stat = ["Normal", "Suspect", "Pathologic"]
        self.start_index = 0
        self.end_index = 1000
        self.max_index = None
        self.edge = None
        
        self.metrics = []

        self.next_button.clicked.connect(self.next_update)
        self.previous_button.clicked.connect(self.previous_update)
        self.combo_box_of_files.currentTextChanged.connect(self.new_data)

        self.new_data()
        self.update_plot()
    

    def new_data(self):
        self.metrics.clear()
        name = str(1000+int(self.combo_box_of_files.currentText()))
        self.fhr_signal, self.uc_signal = extract_data(name)
        self.fhr_signal, self.uc_signal = preprocess_signals(self.fhr_signal, self.uc_signal, 4)
        self.features = calculate_features(self.fhr_signal, self.uc_signal)
        self.max_index = len(self.fhr_signal)-1
        self.edge = self.max_index%1000
        self.update_table()
        self.update_plot()

    def update_table(self):
        global scaler
        feature = np.array(self.features).reshape(1, -1)
        x_test = pd.DataFrame(feature, columns=self.columns_name)
        x_test_scaled = scaler.transform(x_test)
        state = self.model.predict(x_test)[0]
        fetal_state = self.fatel_stat[state]
        self.info_table.setItem(0,0, QTableWidgetItem(fetal_state))
        for i in range(len(self.features)):
            self.info_table.setItem(0,i+1, QTableWidgetItem(f"{self.features[i]:.2f}"))

    
    def update_plot(self):
        if self.start_index >= 0 and self.end_index <= self.max_index:
            self.figure_plot.axes_FHR.clear()
            self.figure_plot.axes_FHR.plot(self.fhr_signal[self.start_index:self.end_index], label='FHR', color='blue')
            self.figure_plot.axes_FHR.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            self.figure_plot.axes_FHR.set_xticks([0,200,400,600,800,1000])
            self.figure_plot.axes_FHR.set_xticklabels([str(self.start_index), str(self.end_index-800),str(self.end_index-600),
                                                      str(self.end_index-400), str(self.end_index-200), str(self.end_index)])
            self.figure_plot.axes_FHR.set_title("Fetal Heart Rate (FHR)")
            self.figure_plot.axes_FHR.set_ylabel("BPM")

            self.figure_plot.axes_UC.clear()
            self.figure_plot.axes_UC.plot(self.uc_signal[self.start_index:self.end_index], label='Uterine Contractions (UC)', color='green')
            self.figure_plot.axes_UC.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            self.figure_plot.axes_UC.set_xticks([0,200,400,600,800,1000])
            self.figure_plot.axes_UC.set_xticklabels([str(self.start_index), str(self.end_index-800),str(self.end_index-600),
                                                      str(self.end_index-400), str(self.end_index-200), str(self.end_index)])
            self.figure_plot.axes_UC.set_title("Uterine Contractions (UC)")
            self.figure_plot.axes_UC.set_ylabel("Intensity")
            self.figure_plot.axes_UC.set_xlabel("Time (samples)")

            self.figure_plot.draw()

    def next_update(self):
        if self.end_index+1000 > self.max_index and self.end_index+self.edge >self.max_index:
            return
        if self.end_index+1000 > self.max_index :
            self.end_index += self.edge
            self.start_index += self.edge
        else:
            self.end_index += 1000
            self.start_index += 1000

        self.update_plot()

    def previous_update(self):
        if self.end_index == self.max_index:
            self.end_index -= self.edge
            self.start_index -= self.edge
        
        elif self.start_index == 0:
            return
        
        else:
            self.start_index -=1000
            self.end_index -= 1000
        
        self.update_plot()



def extract_data(name="1001"):
    record_name = 'cardiotocography-dataset/'
    record_name+=name
    
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



def calculate_features(FHR, UC, sampling_rate=4, long_term_window=1000):

    def variability(signal, window_size):
        """Calculate variability over sliding windows."""
        return np.std([np.std(signal[i:i+window_size]) for i in range(0, len(signal)-window_size, window_size)])

    features = []

    # 1. FHR Baseline (LB)
    baseline_window = 10 * sampling_rate * 60  # 10 minutes in samples
    features.append(np.mean(FHR[-baseline_window:]) if len(FHR) > baseline_window else np.mean(FHR))

    # 2. Number of Accelerations per second (AC)
    acceleration_threshold = 15  # 15 bpm above baseline
    ac_peaks, _ = find_peaks(FHR, height=features[0] + acceleration_threshold)
    features.append(len(ac_peaks) / len(FHR))

    # 3. Number of Fetal Movements per second (FM)
    movement_threshold = 10  # Smaller threshold for movements
    fm_peaks, _ = find_peaks(FHR, height=features[0] + movement_threshold)
    features.append(len(fm_peaks) / len(FHR))

    # 4. Number of Uterine Contractions per second (UC)
    uc_peaks, _ = find_peaks(UC, height=0.5)
    features.append(len(uc_peaks) / len(UC))

    # 5. Number of Light Decelerations per second (DL)
    light_decel = [1 for i in range(1, len(FHR)) if FHR[i] < FHR[i-1] - 15]
    features.append(len(light_decel) / len(FHR))

    # 6. Number of Severe Decelerations per second (DS)
    severe_decel = [1 for i in range(1, len(FHR)) if FHR[i] < FHR[i-1] - 30]
    features.append(len(severe_decel) / len(FHR))

    prolonged_decel = []
    for i in range(1, len(FHR)):
        # Find the deceleration event
        if FHR[i] < FHR[i-1] - 30:
            # Search for the first occurrence of this event lasting for 2 minutes (2 * sampling_rate * 60)
            previous_decel = np.where(FHR[:i] < FHR[i-1] - 30)[0]
            if len(previous_decel) > 0:
                decel_duration = i - previous_decel[-1]
                if decel_duration > (2 * sampling_rate * 60):  # 2-minute duration
                    prolonged_decel.append(1)

    features.append(len(prolonged_decel) / len(FHR))

    # 8. Percentage of Time with Abnormal Short-Term Variability (ASTV)
    abnormal_stv = np.mean([np.std(FHR[i:i+sampling_rate]) < 5 for i in range(0, len(FHR)-sampling_rate, sampling_rate)])
    features.append(abnormal_stv)

    # 9. Mean Short-Term Variability (MSTV)
    short_term_window = sampling_rate * 60  # 1-minute window in samples
    features.append(variability(FHR, short_term_window))

    # 10. Percentage of Time with Abnormal Long-Term Variability (ALTV)
    long_term_samples = long_term_window * sampling_rate
    abnormal_ltv = np.mean([
        np.std(FHR[i:i+long_term_samples]) < 5 for i in range(0, len(FHR)-long_term_samples, long_term_samples)
    ])
    features.append(abnormal_ltv)

    # 11. Mean Long-Term Variability (MLTV)
    window_size = long_term_samples
    long_term_variability = [
        np.std(FHR[i:i+window_size]) for i in range(0, len(FHR) - window_size + 1, window_size)
    ]
    features.append(np.mean(long_term_variability))

    return features

def model_creator():
    global scaler
    data = pd.read_csv("fetal_health.csv")
    x_data = data.drop("fetal_health", axis=1)
    y_data = data["fetal_health"]
    x_train = scaler.fit_transform(x_data)
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_data)
    return model, x_data.columns

# def model_creator():
#     global scaler
    
#     df = pd.read_csv("fetal_health.csv")
#     x = df.iloc[:, :-1]
#     y = df['fetal_health']
    
#     smote = SMOTE(sampling_strategy='auto') 
#     X_resampled, y_resampled = smote.fit_resample(x, y)

#     X_train = scaler.fit_transform(X_resampled)

#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_resampled)
#     return model, x.columns

def preprocess_signals(fhr_signal, uc_signal, sampling_rate):
    fhr_signal = handle_missing_values(fhr_signal)
    uc_signal = handle_missing_values(uc_signal)

    fhr_signal = low_pass_filter(fhr_signal, cutoff=0.5, fs=sampling_rate)
    uc_signal = low_pass_filter(uc_signal, cutoff=0.5, fs=sampling_rate)

    fhr_signal = clip_outliers(fhr_signal, lower_limit=60, upper_limit=200)
    uc_signal = clip_outliers(uc_signal, lower_limit=0, upper_limit=100)

    return fhr_signal, uc_signal

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, "dark_purple.xml")
    window = main()
    window.show()
    sys.exit(app.exec_())