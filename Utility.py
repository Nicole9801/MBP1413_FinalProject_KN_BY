
import os
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import  cross_val_predict

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier


import joblib
import json
from pycaret.classification import *


def split_data(df, target_col='readmitted', test_size=0.2, val_size=0.2, random_state=42):
    # Split independent and dependent variables
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: Train+Val vs Test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: Train vs Val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, stratify=y_trainval, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# Downsample

def downsample(X_train, y_train):
    train_data = pd.concat([X_train, y_train], axis=1)

    majority_class = train_data[train_data['readmitted'] == 0]
    minority_class = train_data[train_data['readmitted'] == 1]

    majority_downsampled = resample(
        majority_class,
        replace=False,  # sample without replacement
        n_samples=len(minority_class),
        random_state=42
    )

    train_downsampled = pd.concat([majority_downsampled, minority_class])

    train_downsampled = train_downsampled.sample(frac=1, random_state=28).reset_index(drop=True)

    X_train_down = train_downsampled.drop(columns=['readmitted'])
    y_train_down = train_downsampled['readmitted']

    return X_train_down, y_train_down


def evaluate_model(y_true, y_pred, y_prob, name):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prc = average_precision_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    return {
        "Model": name,
        "Accuracy": acc,
        'Precision': prec,
        "Sensitivity": rec,
        "Specificity": specificity,
        "F1 Score": f1,
        "AUROC": auc,
        "AUPRC": prc
    }


def train_log_reg(X_train, y_train, X_val, y_val, evaluate_model_fn, class_weight = None, model_name="Logistic Regression"):
  """
  Function to train a logistic regression model, serving as our baseline model
  """
  log_reg = Pipeline([
    ('scaler', StandardScaler()), 
    ('clf', LogisticRegression(max_iter=1000,
                               class_weight= class_weight, 
                               random_state=42
                               )
    )
  ])
  log_reg.fit(X_train, y_train)
  y_log_prob = log_reg.predict_proba(X_val)[:, 1]

  fpr, tpr, thresholds = roc_curve(y_val, y_log_prob)
  youden_j = tpr - fpr
  best_index = np.argmax(youden_j)
  optimal_threshold = thresholds[best_index]


  # Evaluate the model
  y_pred_thres = (y_log_prob >= optimal_threshold).astype(int)

  evaluation_results = evaluate_model(y_val, y_pred_thres, y_log_prob, model_name)
  evaluation_results['Threshold'] = optimal_threshold

  return log_reg, evaluation_results



def train_random_forest(X_train, y_train, X_val, y_val, evaluate_model, class_weight = None,  model_name="Random Forest"):
    """
    Function to train Random Forest with GridSearchCV, finds optimal threshold
    and evaluates the model performance, feature importance

    Returns:
        best random forest model, optimal_threshold (float), evaluation_results, feature importance ranking
    """

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }

    rf = RandomForestClassifier(random_state=42, class_weight= class_weight)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        verbose=0,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    y_prob = best_rf.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = [f1_score(y_val, (y_prob >= t).astype(int)) for t in thresholds]
    best_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_index]

    # Apply threshold and get predictions
    y_pred = (y_prob >= optimal_threshold).astype(int)

    evaluation_results = evaluate_model(y_val, y_pred, y_prob, model_name)
    evaluation_results['Threshold'] = optimal_threshold
    evaluation_results['Best Params'] = grid_search.best_params_

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_rf.feature_importances_
    }).set_index("Feature")

    return best_rf, evaluation_results, importance_df



def train_xgboost_model(X_train, y_train, X_val, y_val, evaluate_model, model_name="XGBoost"):
    """
    Function to train XGBoost with hyperparameter tuning with GridSearchCV,
    finds optimal threshold using ROC (Youden's J), evaluates performances

    Returns:
        best_model (fitted XGBClassifier), optimal_threshold (float), evaluation_results (dict)
    """

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }


    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        verbose=0,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_

    y_prob = best_xgb.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = [f1_score(y_val, (y_prob >= t).astype(int)) for t in thresholds]
    best_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_index]


    # Predict using optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)

    # Evaluate model
    evaluation_results = evaluate_model(y_val, y_pred, y_prob, model_name)
    evaluation_results['Threshold'] = optimal_threshold
    evaluation_results['Best Params'] = grid_search.best_params_

    # Generate feature importance
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_xgb.feature_importances_
    }).set_index("Feature")

    return best_xgb, evaluation_results, importance_df



def train_mlp(X_train, y_train, X_val, y_val, evaluate_model_fn, model_name="MLP + GridSearch"):
    """
    Trains an MLP model with GridSearchCV and evaluates on validation set.
    """

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(max_iter=500, early_stopping=True, random_state=42))
    ])

    param_grid = {
        'clf__hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'clf__alpha': [0.0001, 0.001, 0.01],  # L2 regularization strength
        'clf__learning_rate_init': [0.001, 0.01],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_val_prob = best_model.predict_proba(X_val)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    optimal_threshold = thresholds[best_index]

    y_val_pred = (y_val_prob >= optimal_threshold).astype(int)

    evaluation_results = evaluate_model_fn(y_val, y_val_pred, y_val_prob, model_name)
    evaluation_results['Threshold'] = optimal_threshold
    evaluation_results['Best Params'] = grid_search.best_params_

    return best_model, evaluation_results

