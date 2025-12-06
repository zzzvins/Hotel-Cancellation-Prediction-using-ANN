# Hotel Cancellation Prediction using SuperSAB-ANN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)
![Algorithm](https://img.shields.io/badge/Algorithm-SuperSAB%20Backprop-orange)

A "from-scratch" implementation of a Multi-Layer Perceptron (MLP) Neural Network designed to predict hotel reservation cancellations. This project avoids high-level deep learning libraries (like TensorFlow or PyTorch) to demonstrate the fundamental mathematics of **Batch Backpropagation** with **Adaptive Learning Rates (SuperSAB)**.

---

## 📖 Project Overview

Revenue management in the hospitality industry is heavily impacted by the unpredictability of booking cancellations. This project utilizes a custom-built Artificial Neural Network (ANN) to classify bookings as either "Canceled" or "Not Canceled."

**Key Highlights:**
* **Pure NumPy Implementation:** All matrix operations, gradients, and activation functions are implemented mathematically without relying on high-level libraries.
* **SuperSAB Algorithm:** Implements the *Super Self-Adapting Backpropagation* algorithm (Tollenaere, 1990) which adapts individual learning rates ($\epsilon$) and momentum ($\mu$) to accelerate convergence.
* **Scientific Rigor:** Includes reproducibility seeds, gradient normalization (critical fix for large batch size), and automated architecture search.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `preprocess.py` | **ETL Pipeline.** Handles data cleaning, One-Hot Encoding of categorical variables, and Min-Max normalization of numerical features. |
| `hotel_ann.py` | **The Core Engine.** Contains the `NeuralNetwork` class, the SuperSAB update logic, and the CLI-enabled training loop. |
| `compare_nodes.py` | **Experimental Script.** Runs a comparative analysis across multiple hidden layer sizes (5 to 100 nodes) to empirically determine the optimal topology. |
| `processed_hotel_data.csv` | **Clean Data.** The processed dataset ready for training (inputs normalized 0-1). |
| `Predicting...Cancellation.csv`| **Raw Data.** The original dataset used as input for the preprocessing script. |

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.x installed. The project relies on the following standard data science libraries:
* **NumPy:** For matrix operations.
* **Pandas:** For data manipulation.
* **Scikit-Learn:** For `MinMaxScaler` (preprocessing only).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/zzzvins/Hotel-Cancellation-Prediction-using-ANN.git](https://github.com/zzzvins/Hotel-Cancellation-Prediction-using-ANN.git)
    cd Hotel-Cancellation-Prediction-using-ANN
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas scikit-learn
    ```

---

## ⚙️ Usage Guide

### 1. Data Preprocessing
The repository includes the raw data necessary to run the entire pipeline.
```bash
python3 preprocess.py
```
Output: Generates processed_hotel_data.csv

### 2. Train the Neural Network
Run the main model. By default, it uses the optimal architecture found during research (30 Hidden Nodes, 5000 Epochs).
```bash
python3 hotel_ann.py
```
**Advanced CLI Usage (Dynamic Parameters):**
You can dynamically adjust hyperparameters directly from the command line:

```bash
python3 hotel_ann.py --nodes <N> --epochs <X>
```
N and X accept any integer value to allow for dynamic hyperparameter tuning

### 3. Run Architecture Experiment
To replicate the sensitivity analysis found in the research report, run the comparison script. This trains 8 distinct networks (5, 10, 15, 20, 30, 40, 50, 100) and outputs a performance table.
```bash
python3 preprocess.py
```
Note: This script takes approximately 40 minutes to complete as it trains multiple models sequentially.

---

## 📊 Performance Results
Based on the fine-grained sensitivity analysis, the optimal network topology was determined to be 30 Hidden Nodes.
| Configuration | Hidden Nodes | Test Accuracy | Training Time (s) |
| :--- | :--- | :--- | :--- |
| **Optimal Peak** | **30** | **81.98%** | **~273s** |
| Baseline (Heuristic) | 15 | 81.87% | ~182s |
| Underfitting | 5 | 81.23% | ~73s |

* **Baseline Accuracy:** 67.24% (Majority Class)
* **Model Improvement:** +14.74%

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Clark Vince Diala**
* **Course:** CMSC 191: Fundamentals of Neural Computing
* **Date:** December 2025
* **Institute:** Institute of Computer Science, UPLB
