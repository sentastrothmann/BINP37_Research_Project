#!/usr/bin/env python3

# Import modules
import pandas as pd
import scanpy as sc
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import scipy.sparse as sp
from tqdm.auto import tqdm
from scipy.stats.contingency import crosstab
import seaborn as sns

adata = sc.read_h5ad('8_Classifiers/Data/Input/scanpy_cp_scAtlas.h5ad')

# Set X and Y
X = adata.X
Y = adata.obs['author_cell_type']

# Create an array of indices for all cells
all_indices = np.arange(X.shape[0])

# Split indices, keeping track of which are train/test
train_idx, test_idx = train_test_split(
    all_indices,
    test_size=0.1,
    stratify=Y,
    random_state=42)

# Use indices to subset X and Y
X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

# Train the random forest classifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]}

inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
clf = RandomForestClassifier(max_features='sqrt', bootstrap=True, oob_score=True, n_jobs=1, random_state=42)

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=inner_cv,
    n_jobs=-1)

grid_search.fit(X_train, Y_train)
best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='weighted')

# Print out the  outcomes
print(f"\nBest Params: {grid_search.best_params_}")
print(f"Accuracy on test set: {acc:.3f}")
print(f"F1 score on test set: {f1:.3f}")
print('\nClassification Report:\n', classification_report(Y_test, Y_pred))

joblib.dump(best_model, '8_Classifiers/InitialModels/best_rf_classifier_scAtlas.pkl')