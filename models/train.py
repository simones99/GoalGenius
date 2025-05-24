import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
from datetime import datetime
from typing import Dict, Any, Tuple

from src.features.build_features import handle_outliers_iqr
from models.config import PARAM_DISTRIBUTIONS, MODEL_CONFIGS
from models.utils import save_results
from models.model_utils import prepare_train_test_split, train_evaluate_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_models(processed_data_path: str = 'data/processed/matches_processed.csv') -> None:
    """Main training pipeline."""
    logger.info("Starting model training pipeline")
    
    # Load data
    logger.info(f"Loading data from {processed_data_path}")
    df = pd.read_csv(processed_data_path)
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    
    # Split data
    logger.info("Splitting data into train/val/test sets")
    train, val, test = prepare_train_test_split(df)
    
    # Prepare features
    feature_cols = ['EloDiff', 'FormDiff', 'H2H', 'IsDerby']
    X_train = train[feature_cols]
    y_train = train['FTResult']
    X_val = val[feature_cols]
    y_val = val['FTResult']
    
    # Initialize results tracking
    results = {}
    best_model = None
    best_score = float('inf')
    
    # Train all models
    for name, config in MODEL_CONFIGS.items():
        logger.info(f"Training {name} model")
        
        # Get model class from string
        if name == 'logistic':
            model_class = LogisticRegression
        elif name == 'random_forest':
            model_class = RandomForestClassifier
        else:  # xgboost
            model_class = xgb.XGBClassifier
            
        try:
            model, metrics, scaler = train_evaluate_model(
                model_class,
                X_train,
                y_train,
                X_val,
                y_val,
                config['params'],
                name
            )
            
            # Add feature names and training size to metrics
            metrics.update({
                'feature_names': feature_cols,
                'train_size': len(X_train),
                'val_size': len(X_val)
            })
            
            results[name] = metrics
            logger.info(f"{name} training complete - Log Loss: {metrics['log_loss']:.4f}")
            
            # Track best model
            if metrics['log_loss'] < best_score:
                best_score = metrics['log_loss']
                best_model = (name, model, scaler)
                
        except Exception as e:
            logger.error(f"Error training {name} model: {str(e)}")
            continue
    
    # Save results and best model
    logger.info("Saving results and best model")
    try:
        save_results(results, best_model)
        logger.info("Training pipeline complete!")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

if __name__ == '__main__':
    train_models()
