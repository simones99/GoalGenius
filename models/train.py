import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
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
    trained_models = {}
    trained_scalers = {}
    best_model = None
    best_score = float('inf')
    
    # Train base models first
    base_models = ['random_forest', 'xgboost', 'logistic']
    for name in base_models:
        config = MODEL_CONFIGS[name]
        logger.info(f"Training {name} model")
        
        # Get model class
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
            
            # Store results
            metrics.update({
                'feature_names': feature_cols,
                'train_size': len(X_train),
                'val_size': len(X_val)
            })
            
            results[name] = metrics
            trained_models[name] = model
            trained_scalers[name] = scaler
            
            logger.info(f"{name} training complete - Log Loss: {metrics['log_loss']:.4f}")
            
            # Track best model
            if metrics['log_loss'] < best_score:
                best_score = metrics['log_loss']
                best_model = (name, model, scaler)
                
        except Exception as e:
            logger.error(f"Error training {name} model: {str(e)}")
            continue
    
    # Train stacking model if we have at least 2 base models
    if len(trained_models) >= 2:
        logger.info("Training stacking model")
        try:
            estimators = [
                (name, model) 
                for name, model in trained_models.items()
                if name != 'logistic'  # Don't use logistic in base models
            ]
            
            # Use logistic regression as final estimator
            final_estimator = LogisticRegression(
                multi_class='multinomial',
                max_iter=10000,
                random_state=42
            )
            
            # Create and train stacking model
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=TimeSeriesSplit(n_splits=5),
                n_jobs=-1
            )
            
            # Scale features for stacking
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            stacking.fit(X_train_scaled, y_train)
            
            # Evaluate stacking model
            y_pred_proba = stacking.predict_proba(X_val_scaled)
            metrics = {
                'log_loss': log_loss(y_val, y_pred_proba),
                'accuracy': accuracy_score(y_val, stacking.predict(X_val_scaled))
            }
            
            results['stacking'] = metrics
            logger.info(f"Stacking model complete - Log Loss: {metrics['log_loss']:.4f}")
            
            # Update best model if stacking performs better
            if metrics['log_loss'] < best_score:
                best_score = metrics['log_loss']
                best_model = ('stacking', stacking, scaler)
                
        except Exception as e:
            logger.error(f"Error training stacking model: {str(e)}")
    
    # Save results and best model
    logger.info("Saving results and best model")
    try:
        save_results(results, best_model)
        logger.info("Training pipeline complete!")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

if __name__ == '__main__':
    train_models()
