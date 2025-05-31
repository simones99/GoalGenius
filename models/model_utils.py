import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    log_loss, accuracy_score, f1_score,
    precision_score, recall_score
)
from typing import Dict, Any, Tuple
from models.config import PARAM_DISTRIBUTIONS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.features.build_features import handle_outliers_iqr
from models.utils import save_results

def prepare_train_test_split(df: pd.DataFrame, 
                           test_months: int = 6,
                           val_months: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare time-based train/validation/test splits.
    
    Args:
        df: DataFrame with match data
        test_months: Number of months for test set
        val_months: Number of months for validation set
        
    Returns:
        train, val, test DataFrames
    """
    # Sort by date to ensure temporal ordering
    df = df.sort_values('MatchDate')
    
    # Define cutoff dates
    last_date = df['MatchDate'].max()
    test_cutoff = last_date - pd.DateOffset(months=test_months)
    val_cutoff = test_cutoff - pd.DateOffset(months=val_months)
    
    # Split the data
    train = df[df['MatchDate'] <= val_cutoff].copy()
    val = df[(df['MatchDate'] > val_cutoff) & (df['MatchDate'] <= test_cutoff)].copy()
    test = df[df['MatchDate'] > test_cutoff].copy()
    
    print(f"Training data from {train['MatchDate'].min()} to {train['MatchDate'].max()}")
    print(f"Validation data from {val['MatchDate'].min()} to {val['MatchDate'].max()}")
    print(f"Test data from {test['MatchDate'].min()} to {test['MatchDate'].max()}")
    
    return train, val, test

def train_evaluate_model(model_class: Any,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: pd.DataFrame,
                        y_val: pd.Series,
                        model_params: Dict[str, Any],
                        model_type: str) -> Tuple[Any, Dict[str, float], Any]:
    """Train and evaluate a model with enhanced validation and class balancing."""
    # Initialize scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Handle XGBoost specific parameters
    fit_params = {}
    if model_type == 'xgboost':
        early_stopping_rounds = model_params.pop('early_stopping_rounds', 50)
        fit_params = {
            'eval_set': [(X_val_scaled, y_val)],
            'early_stopping_rounds': early_stopping_rounds,
            'verbose': False
        }
        
        # Calculate class weights for imbalanced data
        if model_params.get('scale_pos_weight') is None:
            class_counts = np.bincount(y_train)
            # For multiclass, we'll use the average ratio of other classes
            weights = [sum(class_counts) / (len(class_counts) * count) for count in class_counts]
            model_params['scale_pos_weight'] = np.mean(weights)
    
    # Initialize and train model
    model = model_class(**model_params)
    model.fit(X_train_scaled, y_train, **fit_params)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_val_scaled)
    y_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    metrics = {
        'log_loss': log_loss(y_val, y_pred_proba),
        'accuracy': accuracy_score(y_val, y_pred),
        'f1_macro': f1_score(y_val, y_pred, average='macro'),
        'precision_macro': precision_score(y_val, y_pred, average='macro'),
        'recall_macro': recall_score(y_val, y_pred, average='macro')
    }
    
    return model, metrics, scaler