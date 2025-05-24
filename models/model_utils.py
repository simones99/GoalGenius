import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Tuple
from scipy.stats import randint, uniform
from src.features.build_features import handle_outliers_iqr
from models.config import PARAM_DISTRIBUTIONS, MODEL_CONFIGS
from models.utils import save_results

def prepare_train_test_split(df: pd.DataFrame, 
                           test_size: float = 0.2,
                           val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train, validation and test sets.
    
    Args:
        df: Processed matches DataFrame
        test_size: Proportion of data to use for testing
        val_size: Proportion of remaining data to use for validation
    """
    # Sort by date
    df = df.sort_values('MatchDate')
    
    # Calculate split points
    test_idx = int(len(df) * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    # Split data
    train = df.iloc[:val_idx]
    val = df.iloc[val_idx:test_idx]
    test = df.iloc[test_idx:]
    
    return train, val, test

def train_evaluate_model(model_class: Any,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: pd.DataFrame,
                        y_val: pd.Series,
                        model_params: Dict[str, Any],
                        model_type: str) -> Tuple[Any, Dict[str, float]]:
    """
    Train a model with appropriate preprocessing and evaluation.
    
    Args:
        model_class: The model class to instantiate
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_params: Model hyperparameters
        model_type: Type of model ('logistic', 'random_forest', or 'xgboost')
    """
    # Copy data to avoid modifying original
    X_train_prep = X_train.copy()
    X_val_prep = X_val.copy()
    
    # Initialize preprocessing objects
    scaler = StandardScaler()
    
    if model_type == 'logistic':
        # For logistic regression: handle outliers and scale features
        X_train_prep = handle_outliers_iqr(X_train_prep, method='clip')
        
        # Scale features
        X_train_prep = scaler.fit_transform(X_train_prep)
        X_val_prep = scaler.transform(X_val_prep)
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train_prep, y_train)
        
    else:  # random_forest or xgboost
        # Use RandomizedSearchCV
        n_iter = 100  # number of parameter settings sampled
        cv = TimeSeriesSplit(n_splits=5)
        
        random_search = RandomizedSearchCV(
            model_class(**model_params),
            param_distributions=PARAM_DISTRIBUTIONS[model_type],
            n_iter=n_iter,
            cv=cv,
            scoring='neg_log_loss',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train_prep, y_train)
        model = random_search.best_estimator_
    
    # Get predictions
    X_val_final = X_val_prep
    y_pred_proba = model.predict_proba(X_val_final)
    y_pred = model.predict(X_val_final)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'log_loss': log_loss(y_val, y_pred_proba),
    }
    
    # Add best params for tree-based models
    if model_type in ['random_forest', 'xgboost']:
        metrics['best_params'] = random_search.best_params_
        metrics['best_score'] = random_search.best_score_
    
    return model, metrics, scaler if model_type == 'logistic' else None