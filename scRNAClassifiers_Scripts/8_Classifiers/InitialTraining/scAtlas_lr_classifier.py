#!/usr/bin/env python3

# Import necessary libraries
import scanpy as sc
from sklearn.linear_model import LogisticRegressionCV
from scipy.stats.contingency import crosstab
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import scipy.sparse as sp
import numpy as np
import pandas as pd

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

clf = LogisticRegressionCV(
    Cs=10,                          # Try 10 different values for C
    cv=10,                          # 10-fold cross-validation
    penalty='l2',                   # or 'elasticnet' if using l1_ratio
    solver='saga',                  # saga handles elasticnet and multiclass
    multi_class='multinomial',     # For softmax/multiclass setup
    scoring='accuracy',            # Scoring metric for CV
    max_iter=1000,                 # More iterations for convergence
    n_jobs=-1,                     # Parallel computation
    random_state=42,
    verbose=1)                      # Optional: helps debug convergence

# Fit the model
clf.fit(X_train, Y_train)

# Predict and evaluate
Y_pred = clf.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='weighted')

# Print the outcomes
print(f"\nBest C values per class:\n{clf.C_}")
print(f"Accuracy on test set: {acc:.3f}")
print(f"F1 score on test set: {f1:.3f}")
print('\nClassification Report:\n', classification_report(Y_test, Y_pred))

# Save model
joblib.dump(clf, '8_Classifiers/InitialModels/best_lr_classifier_scAtlas.pkl')