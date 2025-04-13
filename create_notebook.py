import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text = """\
# Heart Disease Prediction - Data Exploration

This notebook explores the UCI Heart Disease dataset to understand its structure, features, and patterns."""

imports = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
import os

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")"""

load_data = """\
try:
    # Download the Heart Disease dataset directly from UCI
    print("Downloading the UCI Heart Disease dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Define column names
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Download and read the data
    response = requests.get(url)
    data = response.text
    
    # Read the data into a DataFrame
    heart_df = pd.read_csv(StringIO(data), names=column_names)
    
    # Replace '?' with NaN
    heart_df = heart_df.replace('?', np.nan)
    
    # Convert columns to numeric
    for col in heart_df.columns:
        heart_df[col] = pd.to_numeric(heart_df[col], errors='coerce')
    
    # Fill missing values with median for numerical columns
    for col in heart_df.columns:
        if heart_df[col].dtype in ['float64', 'int64']:
            heart_df[col] = heart_df[col].fillna(heart_df[col].median())
    
    # Convert target to binary (0 = no disease, 1 = disease)
    heart_df['target'] = (heart_df['target'] > 0).astype(int)
    
    # Create data directory if it doesn't exist
    os.makedirs('../data/raw', exist_ok=True)
    
    # Save the raw dataset
    heart_df.to_csv('../data/raw/heart_disease_data.csv', index=False)
    
    print("Dataset shape:", heart_df.shape)
    print("Dataset columns:", heart_df.columns.tolist())
    print("First 5 rows:")
    display(heart_df.head())
    
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check your internet connection and try again.")"""

basic_info = """\
try:
    # Display basic information about the dataset
    print("\\nBasic information about the dataset:")
    heart_df.info()
    
    # Display summary statistics
    print("\\nSummary statistics:")
    display(heart_df.describe())
    
    # Check for missing values
    print("\\nMissing values in each column:")
    display(heart_df.isnull().sum())
    
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please make sure the dataset was loaded successfully.")"""

target_dist = """\
try:
    # Display the distribution of the target variable
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=heart_df)
    plt.title('Distribution of Heart Disease')
    plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.show()
    
    # Calculate the percentage of each class
    target_counts = heart_df['target'].value_counts(normalize=True) * 100
    print("\\nPercentage of each class:")
    print(f"No Heart Disease (0): {target_counts[0]:.2f}%")
    print(f"Heart Disease (1): {target_counts[1]:.2f}%")
    
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please make sure the dataset was loaded successfully.")"""

feature_analysis = """\
try:
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = heart_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Heart Disease Features')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please make sure the dataset was loaded successfully.")"""

feature_dist = """\
try:
    # Select numerical features for analysis
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Create box plots for numerical features
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numerical_features):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x='target', y=feature, data=heart_df)
        plt.title(f'{feature} by Heart Disease')
    plt.tight_layout()
    plt.show()
    
    # Analyze categorical features
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    plt.figure(figsize=(15, 15))
    for i, feature in enumerate(categorical_features):
        plt.subplot(3, 3, i+1)
        sns.countplot(x=feature, hue='target', data=heart_df)
        plt.title(f'{feature} by Heart Disease')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please make sure the dataset was loaded successfully.")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(imports),
    nbf.v4.new_markdown_cell("## 1. Loading the Dataset\n\nWe'll use the UCI Heart Disease dataset from OpenML."),
    nbf.v4.new_code_cell(load_data),
    nbf.v4.new_markdown_cell("## 2. Basic Dataset Information\n\nLet's examine the basic information about our dataset."),
    nbf.v4.new_code_cell(basic_info),
    nbf.v4.new_markdown_cell("## 3. Target Variable Distribution\n\nLet's examine the distribution of our target variable (presence of heart disease)."),
    nbf.v4.new_code_cell(target_dist),
    nbf.v4.new_markdown_cell("## 4. Feature Analysis\n\nLet's analyze the relationships between features and the target variable."),
    nbf.v4.new_code_cell(feature_analysis),
    nbf.v4.new_markdown_cell("## 5. Feature Distributions by Target\n\nLet's examine how each feature differs between patients with and without heart disease."),
    nbf.v4.new_code_cell(feature_dist),
    nbf.v4.new_markdown_cell("""## 6. Initial Insights

Based on our exploration, we can make some initial observations:

1. Dataset size and structure
2. Feature distributions
3. Correlations with the target variable
4. Potential preprocessing needs

These insights will guide our next steps in data preprocessing and model development.""")
]

# Create the notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Write the notebook
nbf.write(nb, 'notebooks/1_data_exploration.ipynb') 