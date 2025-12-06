"""
================================================================================
PROJECT:   Hotel Cancellation Prediction using Artificial Neural Networks
ALGORITHM: Batch Backpropagation with Adaptive Learning Rates (SuperSAB)
AUTHOR:    Clark Vince Diala
DATE:      December 2025
================================================================================
Description:
  This script implements a multi-layer perceptron (MLP) from scratch using NumPy.
  It utilizes the SuperSAB algorithm for adaptive weight updates to predict
  hotel reservation cancellations based on customer booking data.

USAGE SYNTAX:
  python3 hotel_ann.py --nodes <N> --epochs <X>

EXAMPLES:
  python3 hotel_ann.py                          # Runs with optimal defaults (30 nodes, 5000 epochs)
  python3 hotel_ann.py --nodes 50               # Runs with N=50 hidden nodes
  python3 hotel_ann.py --epochs 10000           # Runs for X=10000 training iterations
  python3 hotel_ann.py --nodes 40 --epochs 2000 # Customizes both parameters

  *NOTE: <N> and <X> accept any integer value to allow for dynamic
         hyperparameter tuning without modifying the source code.
"""

import numpy as np
import pandas as pd
import time
import argparse

# ==========================================
# 1. CONFIGURATION DEFAULTS
# ==========================================
# Reproducibility Seed 
RANDOM_SEED = 42

# Network Architecture Defaults
# Note: 30 Hidden Nodes was determined to be optimal via comparative analysis 
DEFAULT_NODES = 30      
OUTPUT_NODES = 1        # Binary Classification
DEFAULT_EPOCHS = 5000

# Logging Configuration
LOG_INTERVAL = 500      # Print progress every N epochs
MSE_CHECK_SIZE = 2000   # Number of samples to use for quick MSE calculation

# ==========================================
# SUPER-SAB ALGORITHM HYPERPARAMETERS
# ==========================================
# These control how fast the network learns and adapts.

KAPPA = 0.01  # Learning Rate Increment (Speed up if gradient direction matches)
PHI   = 0.7   # Learning Rate Decay (Slow down if gradient direction flips)
THETA = 0.7   # Gradient Smoothing (Averages past gradients to reduce noise)
MU    = 0.95  # Momentum (Keeps the weight updates moving in the same direction)

# ==========================================
# 2. THE NEURAL NETWORK CLASS
# ==========================================
class NeuralNetwork:
    """
    A Feedforward Neural Network implementing Batch Backpropagation 
    with SuperSAB adaptive learning rates.
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize Weights (Random Uniform)
        # We use small random weights (-0.1 to 0.1) to prevent sigmoid saturation.
        np.random.seed(RANDOM_SEED)
        self.w_ih = np.random.uniform(-0.1, 0.1, (self.input_nodes + 1, self.hidden_nodes))
        self.w_ho = np.random.uniform(-0.1, 0.1, (self.hidden_nodes + 1, self.output_nodes))

        # Initialize Adaptive Memory Matrices (SuperSAB State)
        self.c_ih = np.zeros_like(self.w_ih); self.c_ho = np.zeros_like(self.w_ho)
        self.e_ih = np.full_like(self.w_ih, KAPPA); self.e_ho = np.full_like(self.w_ho, KAPPA)
        self.f_ih = np.zeros_like(self.w_ih); self.f_ho = np.zeros_like(self.w_ho)

    def logistic(self, x):
        """Sigmoid activation with clipping for stability."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

    def train_batch(self, inputs, targets):
        """Executes one full epoch of Batch Training."""
        inputs_b = np.c_[np.ones(len(inputs)), inputs]
        
        # Forward Pass
        y = self.logistic(np.dot(inputs_b, self.w_ih))
        y_b = np.c_[np.ones(len(y)), y]
        z = self.logistic(np.dot(y_b, self.w_ho))

        # Backward Pass
        # Error Signal: (Output - Target) * Output * (1 - Output)
        output_errors = (z - targets) * z * (1.0 - z)
        
        # Gradient Calculation (Normalized by Batch Size to prevent explosion)
        d_ho = np.dot(y_b.T, output_errors) / len(inputs)
        
        hidden_errors = np.dot(output_errors, self.w_ho[1:].T)
        hidden_term = hidden_errors * y * (1.0 - y)
        d_ih = np.dot(inputs_b.T, hidden_term) / len(inputs) 

        # Weight Updates
        self._apply_supersab(self.w_ho, d_ho, self.f_ho, self.e_ho, self.c_ho)
        self._apply_supersab(self.w_ih, d_ih, self.f_ih, self.e_ih, self.c_ih)
    
    def _apply_supersab(self, w, d, f, e, c):
        """Strict implementation of the 'SUB change' routine."""
        same_dir = (d * f) > 0
        e[same_dir] += KAPPA
        e[~same_dir] *= PHI
        f[:] = (1.0 - THETA) * d + (THETA * f)
        c[:] = ((1.0 - MU) * -1.0 * e * d) + (MU * c)
        w += c

    def query(self, inputs):
        """Prediction helper."""
        inputs_b = np.c_[np.ones(len(inputs)), inputs]
        y = self.logistic(np.dot(inputs_b, self.w_ih))
        y_b = np.c_[np.ones(len(y)), y]
        return self.logistic(np.dot(y_b, self.w_ho))

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- Parse CLI Arguments ---
    parser = argparse.ArgumentParser(description="Train Neural Network for Hotel Cancellations")
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES, help="Number of hidden neurons (Dynamic Integer)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs (Dynamic Integer)")
    args = parser.parse_args()

    # Assign CLI args to variables
    HIDDEN_NODES = args.nodes
    EPOCHS = args.epochs
    OUTPUT_NODES = 1 

    print("\n" + "="*60)
    print("   HOTEL CANCELLATION PREDICTION SYSTEM (ANN)")
    print("="*60)

    # 1. Load Data
    data_path = "processed_hotel_data.csv"
    try:
        print(f"[INFO] Loading dataset: {data_path}...")
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"[ERROR] File '{data_path}' not found. Run preprocess.py first.")
        exit()

    # 2. Prepare Data (80/20 Split)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    
    split_idx = int(len(df) * 0.8)
    train_X, test_X = X[:split_idx], X[split_idx:]
    train_y, test_y = y[:split_idx], y[split_idx:]

    INPUT_NODES = train_X.shape[1]

    # 3. Print Configuration
    print("-" * 60)
    print(f" EXPERIMENTAL SETUP")
    print("-" * 60)
    print(f" Input Features:      {INPUT_NODES}")
    print(f" Hidden Neurons:      {HIDDEN_NODES} (Default: {DEFAULT_NODES})")
    print(f" Output Neurons:      {OUTPUT_NODES}")
    print(f" Training Samples:    {len(train_X)}")
    print(f" Test Samples:        {len(test_X)}")
    print(f" Training Epochs:     {EPOCHS}")
    print(f" Algorithm:           SuperSAB (Kappa={KAPPA}, Mu={MU})")
    print("-" * 60)

    # 4. Initialize Network
    nn = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)

    # 5. Training Loop
    print(f"\n[INFO] Starting training process...")
    print(f"{'Epoch':<10} | {'MSE (Loss)':<15}")
    print("-" * 30)

    start_time = time.time()
    for e in range(EPOCHS):
        nn.train_batch(train_X, train_y)
        
        # Log progress based on configuration variable
        if (e+1) % LOG_INTERVAL == 0:
            # Check training error on a subset defined in config
            preds = nn.query(train_X[:MSE_CHECK_SIZE])
            mse = np.mean((train_y[:MSE_CHECK_SIZE] - preds) ** 2)
            print(f"{e+1:<10} | {mse:.6f}")

    training_time = time.time() - start_time
    print("-" * 30)
    print(f"[INFO] Training complete in {training_time:.2f} seconds.")

    # 6. Evaluation
    print("\n" + "="*60)
    print("   FINAL PERFORMANCE EVALUATION")
    print("="*60)
    
    raw_outputs = nn.query(test_X)
    predictions = (raw_outputs > 0.5).astype(int)
    correct = (predictions == test_y).sum()
    accuracy = correct / len(test_y)
    
    print(f" Test Accuracy:       {accuracy * 100:.2f}%")
    print(f" Correct Predictions: {correct} / {len(test_y)}")
    print("="*60 + "\n")