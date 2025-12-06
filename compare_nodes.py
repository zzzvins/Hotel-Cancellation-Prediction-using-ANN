"""
================================================================================
PROJECT:   Hotel Cancellation Prediction using Artificial Neural Networks
COURSE:    CMSC 191: Fundamentals of Neural Computing
EXPERIMENT: FINE-GRAINED SENSITIVITY ANALYSIS OF HIDDEN LAYER TOPOLOGY
AUTHOR:     Clark Vince Diala
DATE:       December 2025
================================================================================
DESCRIPTION:
    Performs a comparative analysis of neural network performance across varying
    hidden layer capacities. This script trains multiple instances of the
    NeuralNetwork class to empirically determine the optimal topology,
    identifying underfitting and overfitting thresholds.
"""

import numpy as np
import pandas as pd
import time
from hotel_ann import NeuralNetwork

# --- Experiment Configuration ---
DATA_FILE = "processed_hotel_data.csv"
NODES_TO_TEST = [5, 10, 15, 20, 30, 40, 50, 100]
EPOCHS_PER_TEST = 5000
RANDOM_SEED = 42

if __name__ == "__main__":
    # --- START ---
    print("\n" + "="*80)
    print(f"{'FINE-GRAINED SENSITIVITY ANALYSIS OF HIDDEN LAYER TOPOLOGY':^80}")
    print("="*80)
    
    print(f"{' EXPERIMENTAL CONFIGURATION':<80}")
    print("-" * 80)
    print(f"  > Architectures Tested  : {len(NODES_TO_TEST)} Configurations")
    print(f"  > Hidden Node Sequence  : {NODES_TO_TEST}")
    print(f"  > Training Epochs       : {EPOCHS_PER_TEST} per configuration")
    print(f"  > Reproducibility Seed  : {RANDOM_SEED}")
    
    # Dynamic Time Estimation
    # (Based on approx 40s per 1000 epochs, adjusted for architecture size)
    print(f"  > Estimated Runtime     : ~40 Minutes")
    print("-" * 80 + "\n")

    # 1. Load and Split Data
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please run preprocess.py first.")
        exit()

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    
    # Standard 80/20 Hold-out Split
    split = int(len(df) * 0.8)
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    results = []

    # 2. Execution Loop
    print(f"{'PROGRESS LOG':<80}")
    print("-" * 80)
    
    for i, hidden_nodes in enumerate(NODES_TO_TEST):
        # Clean, aligned status message
        status_msg = f"Testing Configuration {i+1}/{len(NODES_TO_TEST)}: [ Hidden Nodes: {hidden_nodes} ]"
        print(f"{status_msg:<60}", end="", flush=True)
        
        # Reset seed ensures each architecture starts from the same initialization state
        np.random.seed(RANDOM_SEED)
        start_time = time.time()
        
        # Initialize Network (Imported Class)
        input_nodes = train_X.shape[1]
        nn = NeuralNetwork(input_nodes, hidden_nodes, 1)
        
        # Training Phase
        for e in range(EPOCHS_PER_TEST):
            nn.train_batch(train_X, train_y)

        # Testing Phase
        raw_outputs = nn.query(test_X)
        predictions = (raw_outputs > 0.5).astype(int)
        acc = (predictions == test_y).mean() * 100
        duration = time.time() - start_time
        
        # Print result cleanly
        print(f"-> Done ({duration:.1f}s)")
        results.append((hidden_nodes, acc, duration))

    # 3. Comparative Analysis Report
    print("\n" + "="*80)
    print(f"{'FINAL EXPERIMENTAL RESULTS':^80}")
    print("="*80)
    print(f"{'Nodes':<10} | {'Accuracy':<15} | {'Time':<12} | {'Trend Analysis'}")
    print("-" * 80)
    
    # Calculate statistical bounds for dynamic labeling
    best_acc = max(r[1] for r in results)
    worst_acc = min(r[1] for r in results)
    
    for nodes, acc, dur in results:
        note = "Intermediate"
        
        # Determine performance characteristics relative to the dataset
        if acc == best_acc:
            note = "** OPTIMAL PEAK **"
        elif acc == worst_acc:
            note = "Lowest Accuracy"
        elif nodes == max(NODES_TO_TEST) and acc < best_acc:
            note = "Diminishing Returns"
        elif (best_acc - acc) < 0.2:
            note = "Competitive"
        
        # Output formatted row
        print(f"{nodes:<10} | {acc:.2f}%          | {dur:.0f}s          | {note}")
    print("="*80 + "\n")