# CTG-Heart-Failure-Detection

## Overview
This application uses Cardiotocography (CTG) data to detect potential signs of heart failure in fetuses. It leverages advanced signal processing and machine learning algorithms to analyze fetal heart rate (FHR) and uterine contraction (UC) data, identifying normal, suspect, and pathological conditions.

## Features
- **Data Loading**: Supports `.dat` and `.hea` CTG file formats.
- **Preprocessing**: Cleans and normalizes data for analysis.
- **Feature Extraction**: Derives critical features such as:
  - Baseline FHR and variability.
  - Accelerations and decelerations.
  - Uterine contraction intensity and frequency.
  - FHR response to uterine contractions.
- **Classification**: Uses machine learning to classify data into:
  - Normal
  - Suspect
  - Pathological
- **Interactive Visualization**: Displays CTG signals and feature distributions.
- **Report Generation**: Summarizes results in a user-friendly format.

## Installation

### Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install wfdb numpy pandas scikit-learn matplotlib seaborn
  ```

### Steps
1. Clone the repository:
   ```bash
   https://github.com/Ahmed-Hajhamed/CTG-Heart-Failure-Detection
   ```
2. Navigate to the project directory:
   ```bash
   cd CTG-Heart-Failure-Detection
   ```
3. Run the application:
   ```bash
   python main.py
   ```

## Usage
1. Place `.dat` and `.hea` files in the `/data` folder.
2. Launch the application and select the files for analysis.
3. View the extracted features and classification results.
4. Export results as a report if needed.

## Dataset
We used publicly available CTG datasets from [PhysioNet](https://physionet.org) and other sources. Ensure compliance with dataset licensing when using this application for research or commercial purposes.

## Algorithms
### Preprocessing
- Signal normalization to handle variations in amplitude.
- Noise filtering using Butterworth filters.

### Feature Extraction
- Statistical features: mean, standard deviation, min/max, etc.
- Temporal features: duration of accelerations/decelerations.
- Interaction analysis: correlation between FHR and UC.

### Classification
- Random Forest classifier with hyperparameter tuning using Bayesian optimization.

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Submit a pull request with a detailed description.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [PhysioNet](https://physionet.org) for providing CTG datasets.
- Contributors and community support for development and testing.

## Contact
For questions or feedback, please email ahmed.hajhamed03@eng-st.cu.edu.eg.

