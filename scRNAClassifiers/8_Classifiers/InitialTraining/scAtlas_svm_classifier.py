#!/usr/bin/env python3

# Import modules
import scanpy as sc
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
import matplotlib.pyplot as plt

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

# Define parameter grid
param_grid = {
    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 'auto', 1e-3, 1e-2, 1e-1], 
    'svc__decision_function_shape': ['ovr', 'ovo'],
    'svc__probability': [True]}

# Use a pipeline to standardize and apply SVC
pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('svc', SVC(random_state=42))])

# Cross-validation setup
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV for SVC
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=2)

# Fit the model
grid_search.fit(X_train, Y_train)

# Evaluate
best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='weighted')

# Print the outcomes
print(f"Best SVM Params: {grid_search.best_params_}")
print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}")
print('Classification Report:\n', classification_report(Y_test, Y_pred))

# Save the model
joblib.dump(best_model, '8_Classifiers/InitialModels/best_svm_classifier_scAtlas.pkl')