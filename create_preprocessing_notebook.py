import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text = """\
# Heart Disease Prediction - Data Preprocessing

This notebook focuses on data preprocessing steps including data cleaning, feature engineering, and preparation for model training."""

imports = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")"""

load_data = """\
# Load the dataset
print("Loading the heart disease dataset...")
heart_df = pd.read_csv('../data/raw/heart_disease_data.csv')

# Display basic information
print("\\nDataset shape:", heart_df.shape)
print("\\nFirst 5 rows:")
display(heart_df.head())

# Check for missing values
print("\\nMissing values in each column:")
display(heart_df.isnull().sum())"""

feature_engineering = """\
# Create age groups
heart_df['age_group'] = pd.cut(heart_df['age'], 
                              bins=[0, 40, 55, 65, 100],
                              labels=['<40', '40-55', '55-65', '>65'])

# One-hot encode the age group feature
age_dummies = pd.get_dummies(heart_df['age_group'], prefix='age_group')
heart_df = pd.concat([heart_df, age_dummies], axis=1)
heart_df.drop('age_group', axis=1, inplace=True)

# Create a BMI-like feature (using weight proxy)
# Note: This is a simplified version since we don't have height/weight
heart_df['bmi_proxy'] = heart_df['chol'] / (heart_df['age'] * 0.1)

# Create a blood pressure risk feature
heart_df['bp_risk'] = np.where(
    (heart_df['trestbps'] > 140) | (heart_df['thalach'] < 60),
    1,  # High risk
    0   # Low risk
)

# Display the updated dataset
print("\\nDataset after feature engineering:")
print("New shape:", heart_df.shape)
print("\\nFirst 5 rows:")
display(heart_df.head())"""

data_scaling = """\
# Separate features and target
X = heart_df.drop('target', axis=1)
y = heart_df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'bmi_proxy']

# Fit the scaler on the training data and transform both sets
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Display the scaled training data
print("\\nScaled training data (first 5 rows):")
display(X_train.head())"""

save_data = """\
# Create processed data directory if it doesn't exist
os.makedirs('../data/processed', exist_ok=True)

# Save the preprocessed data
X_train.to_csv('../data/processed/X_train.csv', index=False)
X_test.to_csv('../data/processed/X_test.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)

print("\\nPreprocessed data saved to '../data/processed/' directory")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(imports),
    nbf.v4.new_markdown_cell("## 1. Loading the Dataset\n\nLet's load the dataset we saved from the exploration phase."),
    nbf.v4.new_code_cell(load_data),
    nbf.v4.new_markdown_cell("## 2. Feature Engineering\n\nLet's create some new features that might help our model."),
    nbf.v4.new_code_cell(feature_engineering),
    nbf.v4.new_markdown_cell("## 3. Data Scaling and Splitting\n\nLet's scale our features and split the data into training and testing sets."),
    nbf.v4.new_code_cell(data_scaling),
    nbf.v4.new_markdown_cell("## 4. Saving Preprocessed Data\n\nLet's save our preprocessed data for model training."),
    nbf.v4.new_code_cell(save_data),
    nbf.v4.new_markdown_cell("""## 5. Preprocessing Summary

We have completed the following preprocessing steps:

1. Loaded the raw dataset
2. Created new features:
   - Age groups
   - BMI proxy
   - Blood pressure risk
3. Split the data into training and testing sets
4. Scaled numerical features
5. Saved the preprocessed data

The data is now ready for model training.""")
]

# Create the notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Write the notebook
nbf.write(nb, 'notebooks/2_data_preprocessing.ipynb') 