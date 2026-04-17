#!/usr/bin/env python3

# Import necessary libraries
import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import scipy.sparse as sp
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from scipy.stats.contingency import crosstab

adata = sc.read_h5ad('8_Classifiers/Data/Input/scanpy_cp_scAtlas.h5ad')

# Set X and Y
X = adata.X
Y = adata.obs['author_cell_type']
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

# Create an array of indices for all cells
all_indices = np.arange(X.shape[0])

# Split indices, keeping track of which are train/test
train_idx, test_idx = train_test_split(
    all_indices,
    test_size=0.1,
    stratify=Y_encoded,
    random_state=42)

# Use indices to subset X and Y
X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y_encoded[train_idx], Y_encoded[test_idx]

# Define the function to fit the best estimator
def fit_and_score(estimator, X_train, X_test, Y_train, Y_test):

    estimator.fit(X_train, Y_train, eval_set=[(X_test, Y_test)])

    train_score = estimator.score(X_train, Y_train)
    test_score = estimator.score(X_test, Y_test)

    return estimator, train_score, test_score


# Define classification with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)

clf = xgb.XGBClassifier(tree_method='hist', early_stopping_rounds=3)

results = {}

for train, test in cv.split(X, Y_encoded):
    X_train = X[train]
    X_test = X[test]
    Y_train = Y_encoded[train]
    Y_test = Y_encoded[test]
    est, train_score, test_score = fit_and_score(clone(clf), X_train, X_test, Y_train, Y_test)
    results[est] = (train_score, test_score)

# Find the estimator with the best test score
best_estimator = max(results.items(), key=lambda x: x[1][1])[0]

# Save the best estimator and classifier
joblib.dump(best_estimator, '8_Classifiers/InitialModels/best_xgb_estimator_scAtlas.pkl')
joblib.dump(clf, '8_Classifiers/InitialModels/best_xgb_classifier_scAtlas.pkl')

# Print the outcomes
print('Best model test score:', results[best_estimator][1])
print('Best model train score:', results[best_estimator][0])