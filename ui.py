import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGridLayout, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QLineEdit, QRadioButton,
    QPushButton, QComboBox, QSlider, QFileDialog, QSpacerItem, QSizePolicy, QGraphicsScene, QCheckBox, QTabWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import wfdb


class Figure_CTG(FigureCanvas):
    def __init__(self, parent=None, width=10, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_FHR = fig.add_subplot(211)
        self.axes_UC = fig.add_subplot(212)

        super().__init__(fig)

class ui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beamforming")
        self.setGeometry(100, 100, 1200, 800)
        
        self.fhr_signal, self.uc_signal = extract_data()
        self.start_index = 0
        self.end_index = 1000
        self.max_index = len(self.fhr_signal)-1
        self.edge = self.max_index%1000

        #########################
        self.v_main_layout = QVBoxLayout()
        self.figure_plot = Figure_CTG()

        self.v_main_layout.addWidget(self.figure_plot)
        

        h_layout_of_button = QHBoxLayout()
        self.next_button = QPushButton("NEXT")
        self.next_button.clicked.connect(self.next_update)

        self.previous_button = QPushButton("PREVIOUS") 
        self.previous_button.clicked.connect(self.previous_update)

        h_layout_of_button.addWidget(self.next_button)
        h_layout_of_button.addWidget(self.previous_button)

        self.v_main_layout.addLayout(h_layout_of_button)
        
        #########################
        container = QWidget()
        container.setLayout(self.v_main_layout)
        self.setCentralWidget(container)
        print(self.fhr_signal.shape)
        self.update_plot()

    def update_plot(self):
        print("test 1 ")
        if self.start_index >= 0 and self.end_index <= self.max_index:
            print("test 2 ")
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



def extract_data():
    record_name = 'cardiotocography-dataset/1037'
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ui()
    window.show()
    sys.exit(app.exec_())
