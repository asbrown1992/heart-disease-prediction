import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path

# Load the preprocessed data
X_train = pd.read_csv('../data/processed/X_train.csv')

# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Save the scaler
models_dir = Path('../models')
models_dir.mkdir(exist_ok=True)

scaler_path = models_dir / 'scaler.pkl'
joblib.dump(scaler, scaler_path)

print(f"Scaler saved to {scaler_path}") 