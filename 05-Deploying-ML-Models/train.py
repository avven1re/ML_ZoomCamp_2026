import pandas as pd 
import numpy as np 
import random
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.pipeline import make_pipeline

Base_DIR = Path(__file__).resolve().parent
DATA_PATH = Base_DIR.parent / "03-ML_for_Classification" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_data():
    # Import data
    
    df = pd.read_csv(DATA_PATH)


    # Dataset pre-processing
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'yes').astype(int)
    return df


def train_model(df):
    numerical = ['tenure', 'monthlycharges', 'totalcharges']

    categorical = [
        'gender',
        'seniorcitizen',
        'partner',
        'dependents',
        'phoneservice',
        'multiplelines',
        'internetservice',
        'onlinesecurity',
        'onlinebackup',
        'deviceprotection',
        'techsupport',
        'streamingtv',
        'streamingmovies',
        'contract',
        'paperlessbilling',
        'paymentmethod',
    ]
    y_train = df.churn
    train_dict = df[categorical + numerical].to_dict(orient = 'records')
    pipeline = make_pipeline(
        DictVectorizer(sparse=False), 
        LogisticRegression(solver='liblinear')
    )
    pipeline.fit(train_dict, y_train)
    return pipeline


def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)

df = load_data()
pipeline = train_model(df)
save_model(pipeline, 'model.bin')

print('Model saved to model.bin')
    
#End of this page.