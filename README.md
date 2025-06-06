
# EEG-Based Cognitive State Recognition Using P300

## Overview

This project implements a pipeline for cognitive state recognition using EEG (Electroencephalography) data, focusing on the P300 event-related potential (ERP). The goal is to classify brain responses to standard and oddball stimuli, which can be used for applications like assisting elderly individuals with memory impairments. The pipeline includes data preprocessing, feature extraction, and classification using machine learning models.

The project is implemented in a Jupyter notebook (`eeg_p300_classification.ipynb`) and uses Python libraries like MNE-Python for EEG processing and Scikit-learn for machine learning.

## Features

- **Data Preprocessing**: Cleans EEG data using bandpass filtering and Independent Component Analysis (ICA) to remove artifacts.
- **Feature Extraction**: Extracts P300 amplitude, latency, N200 amplitude, power spectral density (PSD), and wavelet features.
- **Classification**: Uses Support Vector Machine (SVM), Random Forest, and a Voting Classifier to achieve 93% accuracy in classifying standard vs. oddball events.
- **Visualization**: Plots raw EEG, cleaned EEG, ERP waveforms, and confusion matrices for analysis.

## Dataset

The EEG data is sourced from the OpenNeuro dataset `ds003061`, specifically for subject `sub-001`, run 1 (`sub-001_task-P300_run-1_eeg.set`). The dataset includes EEG recordings from a P300 task where participants respond to standard ("beep") and oddball ("boop") stimuli.

- **Source**: OpenNeuro `ds003061`
- **Channels**: 79 channels (65 EEG, 14 auxiliary like EOG, GSR)
- **Sampling Rate**: 256 Hz
- **Events**: 522 standard, 113 oddball, 111 distractor events

## Requirements

To run this project, you need Python 3.x and the following libraries:

- `mne` (for EEG processing)
- `numpy`, `pandas` (for data manipulation)
- `matplotlib`, `seaborn` (for visualization)
- `scikit-learn` (for machine learning)
- `imbalanced-learn` (for SMOTE)
- `pywavelets` (for wavelet transform)
- `awscli` (to download the dataset from OpenNeuro)

### Install the dependencies using:

```bash
pip install mne numpy pandas matplotlib seaborn scikit-learn imbalanced-learn pywavelets awscli
````

## Usage

### Clone the Repository

```bash
git clone https://github.com/your-username/eeg-p300-classification.git
cd eeg-p300-classification
```

### Open the Jupyter Notebook

Launch Jupyter Notebook and open `eeg_p300_classification.ipynb`:

```bash
jupyter notebook
```

### Run the Notebook

The notebook is divided into two main cells:

* **Cell 1**: Downloads the EEG data, preprocesses it, and creates epochs for analysis.
* **Cell 2**: Extracts features, balances the dataset, trains classifiers, and evaluates performance.

Run each cell sequentially to reproduce the results.

### Outputs

Plots are saved in the project directory:

* `raw_eeg_plot.png`: Raw EEG signals.
* `cleaned_eeg_plot.png`: Cleaned EEG signals after preprocessing.
* `erp_plot.png`: ERP waveforms comparing standard, oddball, and distractor events.
* `confusion_matrix.png`: Confusion matrix for the Voting Classifier.

**Classification accuracy**: 93% using the Voting Classifier (SVM + Random Forest)

## Project Structure

* `eeg_p300_classification.ipynb`: Main Jupyter notebook containing the code.
* `README.md`: This file.
* *(Plots are generated and saved during execution.)*

## Results

* **Preprocessing**: Successfully removed artifacts using ICA and filtered EEG data between 0.5–40 Hz.
* **Feature Extraction**: Extracted P300 amplitude, latency, N200 amplitude, alpha/theta PSD, and wavelet coefficients.
* **Classification**:

  * SVM: 92% accuracy
  * Random Forest: 90% accuracy
  * Voting Classifier: **93% accuracy** (best model)
* **Cross-Validation**: Mean F1-macro score of 0.93 for the Voting Classifier.

## Limitations

* The dataset is imbalanced (more standard events than oddball), addressed using SMOTE.
* Only three channels (Fz, Cz, Pz) are used for feature extraction; more channels could improve performance.
* ICA components for artifact removal were manually selected; automated selection could be explored.

## Future Work

* Incorporate more EEG channels for feature extraction.
* Explore deep learning models (e.g., CNNs) for EEG classification.
* Test the pipeline on additional subjects from the dataset.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

* **OpenNeuro** for providing the EEG dataset.
* **MNE-Python** and **Scikit-learn** communities for their excellent libraries and documentation.

---

*Feel free to contribute or raise issues if you encounter any problems!*



