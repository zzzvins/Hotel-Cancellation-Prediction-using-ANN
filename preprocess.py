"""
================================================================================
PROJECT:   Hotel Cancellation Prediction using Artificial Neural Networks
COURSE:    CMSC 191: Fundamentals of Neural Computing [Section EF2]
AUTHOR:    Clark Vince Diala
DATE:      December 2025
================================================================================
Description:
  This script prepares the raw dataset for the Neural Network.
  It performs the following operations:
  1. Drops irrelevant columns (IDs).
  2. Encodes binary targets (Canceled/Not Canceled).
  3. Applies One-Hot Encoding to categorical features.
  4. Normalizes numerical features to [0, 1] range (Min-Max Scaling).
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Configuration
INPUT_FILE = "Predicting Hotel Client's Reservation Cancellation.csv"
OUTPUT_FILE = "processed_hotel_data.csv"

def preprocess_data():
    print("\n" + "="*60)
    print("   DATA PREPROCESSING MODULE")
    print("="*60)

    # 1. Load Data
    print(f"[INFO] Loading raw dataset: '{INPUT_FILE}'...")
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File not found. Please ensure '{INPUT_FILE}' is in this folder.")
        return

    df = pd.read_csv(INPUT_FILE)
    initial_shape = df.shape
    print(f" > Raw Data Loaded: {initial_shape[0]} rows, {initial_shape[1]} columns")

    # 2. Clean Data (Drop ID)
    if 'Booking_ID' in df.columns:
        df = df.drop(columns=['Booking_ID'])
        print(" > Dropped 'Booking_ID' column (irrelevant for prediction).")

    # 3. Encode Target Variable
    # Map "Canceled" -> 1, "Not_Canceled" -> 0
    if 'booking_status' in df.columns:
        df['booking_status'] = df['booking_status'].apply(lambda x: 1 if x == 'Canceled' else 0)
        print(" > Encoded Target: 'booking_status' (1=Canceled, 0=Not Canceled).")

    # 4. Feature Engineering: One-Hot Encoding
    # Converts text categories into binary columns
    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    print(f" > Applying One-Hot Encoding to: {categorical_cols}")
    df = pd.get_dummies(df, columns=categorical_cols)

    # 5. Normalization (Min-Max Scaling)
    # Neural networks require inputs between 0 and 1 to avoid sigmoid saturation
    feature_cols = [col for col in df.columns if col != 'booking_status']
    print(f" > Normalizing {len(feature_cols)} numerical features to range [0, 1]...")
    
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 6. Reorder Columns (Features first, Target last)
    final_cols = feature_cols + ['booking_status']
    df = df[final_cols]

    # 7. Save Processed Data
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("-" * 60)
    print("PREPROCESSING COMPLETE")
    print("-" * 60)
    print(f" Final Dataset Shape: {df.shape}")
    print(f" Input Features:      {len(feature_cols)}")
    print(f" Output File:         {OUTPUT_FILE}")
    print("="*60 + "\n")

if __name__ == "__main__":
    preprocess_data()