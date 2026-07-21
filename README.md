# Machine-Learning-in-Python-Loan-Default-Prediction
Which factors are most important when predicting the likelihood of loan default? This project seeks to create a Loan Default Prediction Algorithm 

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
     
Mounted at /content/drive

# 2. Install needed packages
!pip install xgboost shap -q
     

# 3. Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
