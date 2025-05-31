import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from datetime import datetime
from typing import Dict, Any, Tuple, List

from models.time_aware_models import TimeSeriesEnsemble, evaluate_predictions
from models.config import MODEL_CONFIGS
from src.features.build_features import build_features, build_h2h_features
from src.features.advanced_features import (
    add_league_features, 
    add_match_stats_features, 
    add_advanced_form_features,
    create_interaction_features
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare all features with proper temporal awareness."""
    logger.info("Preparing features...")
    
    # Ensure data is sorted by date
    df = df.sort_values('MatchDate')
    
    # Add features progressively
    logger.info("Adding basic features...")
    df = build_features(df, drop_intermediate=False)
    
    logger.info("Adding league features...")
    df = add_league_features(df)
    
    logger.info("Building H2H features...")
    df = build_h2h_features(df)
    
    logger.info("Adding match statistics features...")
    df = add_match_stats_features(df)
    
    logger.info("Adding advanced form features...")
    df = add_advanced_form_features(df)
    
    logger.info("Creating feature interactions...")
    df = create_interaction_features(df)
    
    # Select final feature set
    feature_cols = [
        # Core features
        'EloDiff', 'FormDiff', 'H2H', 'IsDerby',
        
        # Time-based form features
        'Form5Diff', 'Form10Diff',
        'GoalsScoredDiff', 'GoalsConcededDiff',
        
        # League position and history
        'PrevSeasonPosDiff',
        
        # Match statistics (5-match window)
        'ShotsDiff5', 'TargetDiff5', 'CornersDiff5',
        'FoulsDiff5', 'YellowDiff5', 'RedDiff5',
        
        # Advanced metrics
        'HomeAttackThreat5', 'AwayAttackThreat5',
        'HomeDiscipline5', 'AwayDiscipline5',
        'HomeEfficiency5', 'AwayEfficiency5',
        
        # Consistency metrics
        'ShotsConsistencyDiff5',
        'TargetConsistencyDiff5',
        'CornersConsistencyDiff5',
        
        # Normalized differences
        'ShotsNormDiff5',
        'TargetNormDiff5',
        'CornersNormDiff5',
        
        # Interaction features
        'EloFormInteraction',
        'EloForm5Interaction',
        'GoalsEloInteraction',
        'ShotsEloInteraction',
        'DerbyFormInteraction',
        'FormConsistencyInteraction'
    ]
    
    # Verify all features exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    logger.info(f"Selected {len(feature_cols)} features")
    return df[feature_cols]

def train_ensemble(matches_path: str = 'data/raw/Matches.csv',
                 elo_path: str = 'data/raw/EloRatings.csv') -> None:
    """Main training pipeline using time-aware ensemble."""
    logger.info("Starting ensemble model training pipeline")
    
    # Load raw data
    logger.info("Loading raw data...")
    from src.ingest.github_ingest import load_matches, load_elo
    
    df = load_matches()
    logger.info(f"Loaded {len(df)} matches from {df['MatchDate'].min()} to {df['MatchDate'].max()}")
    
    # Filter out rows with missing target values
    df = df.dropna(subset=['FTResult'])
    
    # Prepare features
    X = prepare_features(df)
    y = df['FTResult'].map({'H': 0, 'D': 1, 'A': 2})
    dates = df['MatchDate']
    
    # Split data temporally
    cutoff_dates = {
        'test': dates.max() - pd.DateOffset(months=3),
        'val': dates.max() - pd.DateOffset(months=6)
    }
    
    train_mask = dates <= cutoff_dates['val']
    val_mask = (dates > cutoff_dates['val']) & (dates <= cutoff_dates['test'])
    test_mask = dates > cutoff_dates['test']
    
    splits = {
        'train': (X[train_mask], y[train_mask], dates[train_mask]),
        'val': (X[val_mask], y[val_mask], dates[val_mask]),
        'test': (X[test_mask], y[test_mask], dates[test_mask])
    }
    
    logger.info(f"Train size: {sum(train_mask)}, Val size: {sum(val_mask)}, Test size: {sum(test_mask)}")
    
    # Create base models
    base_models = [
        (
            'logistic',
            LogisticRegression,
            {
                'multi_class': 'multinomial',
                'max_iter': 5000,
                'solver': 'lbfgs',
                'random_state': 42
            }
        ),
        (
            'random_forest',
            RandomForestClassifier,
            {
                'n_estimators': 500,
                'max_depth': 20,
                'random_state': 42,
                'n_jobs': -1
            }
        ),
        (
            'xgboost',
            xgb.XGBClassifier,
            {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 6,
                'learning_rate': 0.01,
                'n_estimators': 500,
                'random_state': 42,
                'tree_method': 'hist'
            }
        )
    ]
    
    # Train ensemble
    ensemble = TimeSeriesEnsemble(base_models)
    logger.info("Training ensemble model...")
    ensemble.fit(*splits['train'])
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_pred = ensemble.predict(splits['val'][0])
    val_pred_proba = ensemble.predict_proba(splits['val'][0])
    val_metrics = evaluate_predictions(splits['val'][1], val_pred, val_pred_proba)
    
    logger.info(f"Validation metrics:")
    logger.info(f"Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"Log Loss: {val_metrics['log_loss']:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_pred = ensemble.predict(splits['test'][0])
    test_pred_proba = ensemble.predict_proba(splits['test'][0])
    test_metrics = evaluate_predictions(splits['test'][1], test_pred, test_pred_proba)
    
    logger.info(f"Test metrics:")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Log Loss: {test_metrics['log_loss']:.4f}")
    
    # Get feature importance
    feature_importance = ensemble.get_feature_importance()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_importance': feature_importance,
        'ensemble_weights': dict(zip([m[0] for m in base_models], ensemble.weights)),
        'timestamp': timestamp,
        'feature_names': X.columns.tolist()
    }
    
    import json
    import os
    
    # Save metrics
    os.makedirs('data/results', exist_ok=True)
    metrics_path = f'data/results/metrics_ensemble_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save model
    import pickle
    os.makedirs('data/models', exist_ok=True)
    model_path = f'data/models/champion_ensemble_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)
    
    logger.info(f"Results saved to {metrics_path}")
    logger.info(f"Model saved to {model_path}")
    logger.info("Training pipeline complete!")

if __name__ == '__main__':
    train_ensemble()
