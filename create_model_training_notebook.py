import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text = """\
# Heart Disease Prediction - Model Training

This notebook focuses on training and evaluating multiple machine learning models for heart disease prediction."""

imports = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
import pickle
import os

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")"""

load_data = """\
# Load the preprocessed data
print("Loading preprocessed data...")
X_train = pd.read_csv('../data/processed/X_train.csv')
X_test = pd.read_csv('../data/processed/X_test.csv')
y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()

print("\\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)"""

model_training = """\
# Define a function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Print results
    print(f"\\nModel: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model.__class__.__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Plot ROC curve if the model supports predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model.__class__.__name__}')
        plt.legend(loc='lower right')
        plt.show()
    
    return model, accuracy, precision, recall, f1

# Train and evaluate multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\\n{'='*50}")
    print(f"Training and evaluating {name}...")
    print(f"{'='*50}")
    
    model_fitted, accuracy, precision, recall, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {
        'model': model_fitted,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }"""

model_comparison = """\
# Create a DataFrame to compare model performance
performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results],
    'Precision': [results[model]['precision'] for model in results],
    'Recall': [results[model]['recall'] for model in results],
    'F1 Score': [results[model]['f1'] for model in results]
})

# Sort by F1 score (balances precision and recall)
performance_df = performance_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)

# Display the performance comparison
print("\\nModel Performance Comparison:")
display(performance_df)

# Visualize model performance
plt.figure(figsize=(12, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    sns.barplot(x='Model', y=metric, data=performance_df)
    plt.title(f'Model Comparison - {metric}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""

save_model = """\
# Identify the best model based on F1 score
best_model_name = performance_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

print(f"\\nBest performing model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"Precision: {results[best_model_name]['precision']:.4f}")
print(f"Recall: {results[best_model_name]['recall']:.4f}")
print(f"F1 Score: {results[best_model_name]['f1']:.4f}")

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Save the best model
with open('../models/best_heart_disease_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print(f"\\nBest model ({best_model_name}) saved to '../models/best_heart_disease_model.pkl'")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(imports),
    nbf.v4.new_markdown_cell("## 1. Loading Preprocessed Data\n\nLet's load the preprocessed data we saved in the previous notebook."),
    nbf.v4.new_code_cell(load_data),
    nbf.v4.new_markdown_cell("## 2. Model Training and Evaluation\n\nLet's train and evaluate multiple machine learning models."),
    nbf.v4.new_code_cell(model_training),
    nbf.v4.new_markdown_cell("## 3. Model Comparison\n\nLet's compare the performance of all models."),
    nbf.v4.new_code_cell(model_comparison),
    nbf.v4.new_markdown_cell("## 4. Saving the Best Model\n\nLet's save the best performing model for future use."),
    nbf.v4.new_code_cell(save_model),
    nbf.v4.new_markdown_cell("""## 5. Training Summary

We have completed the following steps:

1. Loaded the preprocessed data
2. Trained and evaluated multiple models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine
3. Compared model performance using various metrics
4. Selected and saved the best performing model

The model is now ready for making predictions on new data.""")
]

# Create the notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Write the notebook
nbf.write(nb, 'notebooks/3_model_training.ipynb') 