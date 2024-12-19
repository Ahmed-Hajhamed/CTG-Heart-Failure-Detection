import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QPushButton, QComboBox,
    QHeaderView
)
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import wfdb
gray_style = {
    'axes.facecolor': '2E2E2E',    # Dark gray background for axes
    'figure.facecolor': '31363b',  # Dark gray background for the figure
    'axes.edgecolor': 'white',      # White edges for axes
    'axes.labelcolor': 'white',     # White labels
    'xtick.color': 'white',         # White tick marks on x-axis
    'ytick.color': 'white',         # White tick marks on y-axis
    'text.color': 'white',          # White text
    'grid.color': '#444444',        # Slightly lighter grid lines
    'grid.linestyle': '--',         # Dashed grid lines
    'lines.color': 'cyan',          # Default line color
    'patch.edgecolor': 'white',     # Edge color for patches
    'legend.facecolor': '#4C4C4C',  # Darker gray background for legends
    'legend.edgecolor': 'white',    # White edges for legends
}
plt.style.use(gray_style)

class Figure_CTG(FigureCanvas):
    def __init__(self, parent=None, width=10, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_FHR = fig.add_subplot(211)
        self.axes_UC = fig.add_subplot(212)

        super().__init__(fig)

class ui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CTG Heart Failure Monitoring")
        self.setGeometry(100, 100, 1200, 800)
        
        #########################
        self.v_main_layout = QVBoxLayout()
        
        self.figure_plot = Figure_CTG()
        self.v_main_layout.addWidget(self.figure_plot)
        
        self.info_table = QTableWidget()
        self.info_table.setFixedHeight(100)
        font = QFont("Arial", 36)  # Font family and size
        self.info_table.setFont(font)
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        column_headers = [
            "FETAL STATE",
            "FHR baseline bpm",
            "AC",
            "FM",
            "UC",
            "DL",
            "DS",
            "PD",
            "ASTV",
            "MSTV",
            "ALTV",
            "MLTV",
        ]
        #     # Store results in features dictionary
#     features.append(baseline)
#     features.append(accelerations)
#     features.append(movements)
#     features.append(contractions)
#     features.append(light_decel)
#     features.append(severe_decel)
#     features.append(prolonged_decel)
#     features.append(abnormal_stv)
#     features.append(mstv)
#     features.append(abnormal_ltv)
#     features.append(mltv)
        self.info_table.setRowCount(1)
        self.info_table.setColumnCount(len(column_headers))
        self.info_table.setHorizontalHeaderLabels(column_headers)
        # self.info_table.resizeColumnsToContents()

        self.v_main_layout.addWidget(self.info_table)



        h_layout_of_button = QHBoxLayout()
        self.combo_box_of_files = QComboBox()
        self.combo_box_of_files.addItems([str(i) for i in range(1,507)])
        self.combo_box_of_files.setMaxVisibleItems(10)
        self.combo_box_of_files.setStyleSheet("QComboBox { combobox-popup: 0; }")
        # dropdown_view = self.combo_box_of_files.view()
        # dropdown_view.setMinimumHeight(50)  # Minimum height for dropdown menu
        # dropdown_view.setFixedHeight(150)
        h_layout_of_button.addWidget(self.combo_box_of_files)

        self.next_button = QPushButton("NEXT")
        h_layout_of_button.addWidget(self.next_button)

        self.previous_button = QPushButton("PREVIOUS") 
        h_layout_of_button.addWidget(self.previous_button)

        self.v_main_layout.addLayout(h_layout_of_button)
        
        #########################
        container = QWidget()
        container.setLayout(self.v_main_layout)
        self.setCentralWidget(container)



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
