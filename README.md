# Autism-Prediction-using-Machine-Learning

## Overview

This project aims to predict Autism Spectrum Disorder (ASD) based on responses to a screening questionnaire. The dataset contains responses to screening questions, demographic information, and medical history.

## Dataset

- **File:** `train.csv`
- **Rows:** 800
- **Columns:** 22
- **Target Variable:** `Class/ASD`

### Features:

- `A1_Score` to `A10_Score`: Binary responses to screening questions.
- `age`: Age of the individual.
- `gender`: Gender of the individual.
- `ethnicity`: Ethnic background.
- `jaundice`: History of jaundice at birth.
- `austim`: Family history of autism.
- `contry_of_res`: Country of residence.
- `used_app_before`: Previous use of autism screening app.
- `result`: Screening test score.
- `age_desc`: Age description.
- `relation`: Relation of respondent to the individual.

## Project Workflow

### 1. Importing Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
```

### 2. Data Loading & Understanding

```python
df = pd.read_csv("train.csv")
print(df.shape)
df.head()
```

### 3. Data Preprocessing

- Handling missing values.
- Encoding categorical variables.
- Addressing class imbalance using SMOTE.

### 4. Model Training

- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
- Hyperparameter tuning using RandomizedSearchCV.

### 5. Model Evaluation

- Accuracy Score
- Confusion Matrix
- Classification Report

### 6. Saving the Model

```python
with open("autism_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)
```

## How to Run

1. Install dependencies:
   ```sh
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```
2. Run the Jupyter Notebook step by step.
3. Use `autism_model.pkl` to make predictions.

## Results

The model predicts ASD with good accuracy, leveraging machine learning techniques to enhance diagnosis.



