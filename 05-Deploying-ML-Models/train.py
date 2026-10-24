import pandas as pd 
import numpy as np 
import random
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.pipeline import make_pipeline

# Import dataset
df = pd.read_csv(
    '../03-ML_for_Classification/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )

# Dataset pre-processing
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

## Train test split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

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



## One-Hot_encoding with DictVectorizor
dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

# define func.
def train(df_train, y_train, C = 1.0):
    dicts = df_train[categorical + numerical].to_dict(orient = 'records')
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C = C, max_iter = 3000)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient = 'records')
    
    X = dv.transform(dicts)
    
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

# K-fold CV
# kfold = KFold(n_splits=10, shuffle=True, random_state=1)

# n_splits = 5

# for C in [0.001, 0.01, 0.1, 0.5, 1, 5, 10]:
#     kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

#     scores = []

#     for train_idx, val_idx in kfold.split(df_full_train):
#         df_train = df_full_train.iloc[train_idx]
#         df_val = df_full_train.iloc[val_idx]

#         y_train = df_train.churn.values
#         y_val = df_val.churn.values

#         dv, model = train(df_train, y_train, C=C)
#         y_pred = predict(df_val, dv, model)

#         auc = roc_auc_score(y_val, y_pred)
#         scores.append(auc)

#     print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))
    
C = 1.0
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
#y_pred = predict(df_test, dv, model)

#auc = roc_auc_score(y_test, y_pred)

output_file = f'model_C={C}.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out) # save the model and dictvectorizer for one-hot-encoding
    
#End of this page.