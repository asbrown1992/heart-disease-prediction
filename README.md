# Heart Disease Prediction Model

This project aims to develop a machine learning model that can predict the presence of heart disease based on patient health metrics. The model analyzes various patient attributes such as age, sex, cholesterol levels, and other medical measurements to classify patients as either at risk of heart disease or not.

## Project Structure

```
heart_disease_prediction/
│
├── data/
│   ├── raw/                  # Original dataset files
│   └── processed/            # Cleaned and preprocessed data
│
├── notebooks/
│   ├── 1_data_exploration.ipynb    # Data exploration and visualization
│   ├── 2_data_preprocessing.ipynb  # Data cleaning and feature engineering
│   ├── 3_model_training.ipynb      # Model training and hyperparameter tuning
│   └── 4_model_evaluation.ipynb    # Model evaluation and interpretation
│
├── src/                      # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
│
├── models/                   # Saved model files
│
├── requirements.txt          # Project dependencies
│
└── README.md                 # Project documentation
```

## Dataset

We use the UCI Heart Disease dataset, which contains medical records of 303 patients with 13 different attributes. The dataset includes:

1. **age**: Age in years
2. **sex**: Gender (1 = male, 0 = female)
3. **cp**: Chest pain type (1-4)
4. **trestbps**: Resting blood pressure
5. **chol**: Serum cholesterol
6. **fbs**: Fasting blood sugar
7. **restecg**: Resting electrocardiographic results
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina
10. **oldpeak**: ST depression
11. **slope**: Slope of the peak exercise ST segment
12. **ca**: Number of major vessels
13. **thal**: Thalassemia

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run Jupyter notebooks:
   ```bash
   jupyter notebook
   ```

## Project Workflow

1. **Data Exploration**: Understand the dataset and its features
2. **Data Preprocessing**: Clean the data and prepare it for modeling
3. **Model Training**: Train multiple classification models
4. **Model Evaluation**: Evaluate and compare model performance
5. **Model Interpretation**: Understand what factors influence predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- Scikit-learn, Pandas, and other open-source libraries 