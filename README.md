# Temporal Behavioral Analysis with Adaptive Quantum Neural Network (AQNN) in ZAD-ML

This repository provides a Python implementation of the Temporal Behavioral Analysis pipeline using an Adaptive Quantum Neural Network (AQNN) within the ZAD-ML (Zero Attack Detection in Machine Learning) framework. This system is designed for anomaly detection in multivariate time series data by leveraging quantum circuits, neural networks, and a robust attention mechanism.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Preprocessing Steps](#preprocessing-steps)
- [Model Training and Anomaly Detection](#model-training-and-anomaly-detection)
- [Evaluation](#evaluation)
- [Reproducibility](#reproducibility)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The `Temporal Behavioral Analysis` project focuses on detecting zero-day attacks by identifying anomalies in temporal data patterns. Using quantum circuits and a deep autoencoder network, the AQNN architecture provides robust anomaly detection capabilities by learning adaptive representations of normal behavior patterns.

## Features

- Adaptive Quantum Neural Network (AQNN): Leverages quantum state representations to capture subtle data anomalies.
- Dynamic Attention Mechanism: Enhances the model’s sensitivity to relevant features over time.
- Temporal Segmentation and Adaptive Normalization: Ensures data segments are optimally analyzed based on statistical properties.
- Adaptive Data Augmentation: Adds noise and transformations to the dataset for model robustness.
- Real-time Anomaly Detection: The model dynamically adjusts to data variations, optimizing detection sensitivity with an adaptive threshold.

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/username/ZAD-ML-Temporal-Behavioral-Analysis.git
cd ZAD-ML-Temporal-Behavioral-Analysis
```

### Dependencies

Ensure you have Python 3.7+ and the following libraries installed:

```bash
pip install numpy pandas statsmodels scipy tensorflow==2.8.0 pennylane==0.19.0
```

The provided code is compatible with TensorFlow 2.8.0 and PennyLane 0.19.0. Specific versions are required to ensure consistency in quantum and deep learning computations.

## Usage

### Data Preparation and Preprocessing

The provided code includes an end-to-end preprocessing pipeline (`preprocess()`) that normalizes the dataset, applies intelligent segmentation, and augments data:

```python
# Import necessary libraries and functions
from preprocessing import preprocess

# Example multivariate time series data
data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100)
})

# Run the preprocessing pipeline
preprocessed_data = preprocess(data)
```

### Running Temporal Behavioral Analysis

After preprocessing, initiate the `TemporalBehavioralAnalysis` class to run the analysis on your data:

```python
# Define the input parameters
input_dim = 10  # Input feature vector dimension
encoding_dims = [8, 4, 2]  # Autoencoder configuration
rnn_units = 32
data_stream = np.random.rand(100, input_dim)  # Sample data stream
baseline_hidden_state = np.mean(data_stream, axis=0)  # Baseline for anomaly detection

# Initialize the model
tba_model = TemporalBehavioralAnalysis(input_dim, encoding_dims, rnn_units=rnn_units)

# Run the analysis
tba_model.run_temporal_analysis(data_stream, baseline_hidden_state, verbose=True)
```

### Model Training and Anomaly Detection

The AQNN framework uses quantum state updates to refine anomaly detection capabilities, with anomaly scores and dynamic thresholds outputted for each epoch. This helps identify unusual behavior in the data stream.

## Code Structure

The repository is structured as follows:

- `preprocessing.py`: Contains data normalization, segmentation, feature enhancement, and augmentation functions.
- `model.py`: Defines the AQNN model, including quantum state updates, attention mechanisms, and anomaly scoring.
- `train.py`: Executes the temporal behavioral analysis pipeline, prints progress, and evaluates the model.
- `README.md`: Provides details on the repository, including usage instructions, features, and an example workflow.

## Preprocessing Steps

1. Adaptive Normalization: Normalizes each feature using dynamic scaling factors based on interquartile range (IQR).
2. Intelligent Segmentation: Segments data into windows using an optimal ARIMA-determined window size.
3. Feature Enhancement: Adds statistical features like mean, variance, skewness, and kurtosis to each segment.
4. Adaptive Data Augmentation: Adds noise for robustness and enhances feature variety.

## Model Training and Anomaly Detection

1. Quantum State Update: Encodes data into a quantum state and updates parameters for optimal representation.
2. Attention Mechanism: Computes attention weights on quantum outputs, directing focus to relevant features.
3. Anomaly Score Calculation: Compares hidden states with baseline values, weighted by attention scores.
4. Dynamic Threshold Adaptation: Adjusts the threshold in real time based on data distribution.

## Evaluation

The model provides real-time anomaly scoring and dynamic threshold adjustments to detect anomalies adaptively. Anomalies are flagged when the score exceeds the threshold, which is recalculated based on recent data.

## Reproducibility

The following steps ensure reproducibility:
- Random Seed Setting: Ensures consistent output with `np.random.seed(42)` and `tf.random.set_seed(42)`.
- Explicit Dependency Versions: Requires specific versions of TensorFlow, PennyLane, and other packages.
- Adaptive Mechanisms: Dynamic threshold and real-time updates ensure the model adapts to changes in the data stream.

## Dependencies

- `numpy==1.21.0`
- `pandas`
- `tensorflow==2.8.0`
- `pennylane==0.19.0`
- `statsmodels`
- `scipy`

Install these packages using the following command:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions to enhance the quantum anomaly detection system, model optimizations, or documentation improvements are welcome. Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Example Output

Example log outputs include anomaly scores and adaptive thresholds, providing insight into real-time adjustments:

```plaintext
Epoch 1, Step 0, Anomaly Score: 0.05, Loss: 0.0023
Epoch 1, Step 50, Anomaly Score: 0.10, Loss: 0.0054
Adaptive Threshold at Epoch 1: 0.12

#Final Message
This README outlines the functionality of the ZAD-ML Temporal Behavioral Analysis with AQNN, offering all the steps needed to set up, run, and understand the workflow. Happy ZAD-ML anomaly detection!
