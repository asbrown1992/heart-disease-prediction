import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text = """\
# Heart Disease Prediction - Model Evaluation

This notebook focuses on interpreting the trained model and understanding how it makes predictions."""

imports = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import shap

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")"""

load_model = """\
# Load the trained model
print("Loading the trained model...")
with open('../models/best_heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

print(f"Loaded model: {model.__class__.__name__}")

# Load the test data
X_test = pd.read_csv('../data/processed/X_test.csv')
y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()

print("\\nTest set shape:", X_test.shape)"""

feature_importance = """\
# Function to get feature importance based on model type
def get_feature_importance(model, feature_names):
    if hasattr(model, 'coef_'):
        # For linear models like Logistic Regression
        importance = np.abs(model.coef_[0])
        return pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models like Random Forest
        importance = model.feature_importances_
        return pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    
    else:
        # For models without direct feature importance
        print("This model doesn't provide direct feature importance.")
        return None

# Get feature importance
feature_names = X_test.columns
importance_df = get_feature_importance(model, feature_names)

if importance_df is not None:
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Display feature importance
    print("\\nFeature Importance:")
    display(importance_df)
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Feature Importance - {model.__class__.__name__}')
    plt.tight_layout()
    plt.show()"""

model_performance = """\
# Make predictions on the test set
y_pred = model.predict(X_test)

# Get probability predictions if available
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model.__class__.__name__}')
    plt.legend(loc='lower right')
    plt.show()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model.__class__.__name__}')
    plt.legend(loc='lower left')
    plt.show()"""

shap_analysis = """\
# SHAP analysis for model interpretation
if hasattr(model, "predict_proba"):
    try:
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict_proba, X_test)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.show()
        
        # Detailed SHAP plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test)
        plt.title('SHAP Feature Impact')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"SHAP analysis not available: {str(e)}")"""

prediction_examples = """\
# Create a function to make predictions on new data
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Create a DataFrame with the input features
    data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    # Add the age group features
    age_group = pd.cut([age], bins=[0, 40, 55, 65, 100], labels=['<40', '40-55', '55-65', '>65'])[0]
    for group in ['<40', '40-55', '55-65', '>65']:
        data[f'age_group_{group}'] = 1 if age_group == group else 0
    
    # Create BMI proxy and BP risk features
    data['bmi_proxy'] = data['chol'] / (data['age'] * 0.1)
    data['bp_risk'] = np.where(
        (data['trestbps'] > 140) | (data['thalach'] < 60),
        1,  # High risk
        0   # Low risk
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'bmi_proxy']
    
    # Load the scaler from the training data
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_train_num = X_train[numerical_features]
    scaler.fit(X_train_num)
    
    # Scale the input data
    data[numerical_features] = scaler.transform(data[numerical_features])
    
    # Make prediction
    prediction = model.predict(data)[0]
    
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(data)[0, 1]
        return prediction, probability
    else:
        return prediction, None

# Test the prediction function with example cases
print("\\nExample Predictions:")
print("=" * 50)

# Example 1: Low risk patient
print("\\nExample 1: Low Risk Patient")
pred1, prob1 = predict_heart_disease(
    age=35, sex=0, cp=0, trestbps=120, chol=180, fbs=0,
    restecg=0, thalach=160, exang=0, oldpeak=0.5,
    slope=1, ca=0, thal=3
)
print(f"Heart Disease: {'Yes' if pred1 == 1 else 'No'}")
if prob1 is not None:
    print(f"Probability: {prob1:.4f}")

# Example 2: High risk patient
print("\\nExample 2: High Risk Patient")
pred2, prob2 = predict_heart_disease(
    age=65, sex=1, cp=2, trestbps=160, chol=280, fbs=1,
    restecg=1, thalach=120, exang=1, oldpeak=2.5,
    slope=2, ca=2, thal=6
)
print(f"Heart Disease: {'Yes' if pred2 == 1 else 'No'}")
if prob2 is not None:
    print(f"Probability: {prob2:.4f}")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(imports),
    nbf.v4.new_markdown_cell("## 1. Loading the Model\n\nLet's load the trained model and test data."),
    nbf.v4.new_code_cell(load_model),
    nbf.v4.new_markdown_cell("## 2. Feature Importance\n\nLet's analyze which features are most important for the model's predictions."),
    nbf.v4.new_code_cell(feature_importance),
    nbf.v4.new_markdown_cell("## 3. Model Performance\n\nLet's evaluate the model's performance using various metrics."),
    nbf.v4.new_code_cell(model_performance),
    nbf.v4.new_markdown_cell("## 4. SHAP Analysis\n\nLet's use SHAP values to understand how each feature affects the predictions."),
    nbf.v4.new_code_cell(shap_analysis),
    nbf.v4.new_markdown_cell("## 5. Making Predictions\n\nLet's test the model with some example cases."),
    nbf.v4.new_code_cell(prediction_examples),
    nbf.v4.new_markdown_cell("""## 6. Evaluation Summary

We have completed the following steps:

1. Loaded and analyzed the trained model
2. Identified the most important features
3. Evaluated model performance using various metrics
4. Used SHAP values to understand feature impact
5. Tested the model with example cases

This analysis helps us understand:
- Which factors are most important for heart disease prediction
- How the model makes its predictions
- The model's strengths and limitations
- How to interpret the model's predictions for new patients""")
]

# Create the notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Write the notebook
nbf.write(nb, 'notebooks/4_model_evaluation.ipynb') 