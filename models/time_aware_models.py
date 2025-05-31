import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Tuple, List
from scipy.stats import randint, uniform
from src.features.build_features import handle_outliers_iqr
from models.config import PARAM_DISTRIBUTIONS, MODEL_CONFIGS

class TimeAwareModel:
    """Base class for time-aware models with proper train/validation splitting."""
    
    def __init__(self, model_class: Any, model_params: Dict[str, Any]):
        self.model_class = model_class
        self.model_params = model_params
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def _validate_temporal_order(self, dates: pd.Series) -> bool:
        """Check if data is properly ordered by time."""
        return (dates.diff()[1:] >= pd.Timedelta(0)).all()
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Scale features with option to fit or just transform."""
        # Fill missing values with median
        X = X.fillna(X.median())
        
        # Scale features
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def _handle_outliers(self, X: pd.DataFrame, reference: pd.DataFrame = None) -> pd.DataFrame:
        """Handle outliers with option to use reference data statistics."""
        try:
            if reference is not None:
                return handle_outliers_iqr(X, method='clip', reference_data=reference)
            return handle_outliers_iqr(X, method='clip')
        except Exception as e:
            print(f"Warning: Error in outlier handling: {e}")
            return X
    
    def _compute_class_weights(self, y: pd.Series) -> Dict:
        """Compute balanced class weights."""
        counts = np.bincount(y)
        weights = len(y) / (len(np.unique(y)) * counts)
        return dict(enumerate(weights))
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> None:
        """Fit the model ensuring temporal awareness."""
        if not self._validate_temporal_order(dates):
            raise ValueError("Data must be sorted in chronological order")
        
        self.feature_names = X.columns.tolist()
        
        # Handle outliers and scale features
        X_processed = self._handle_outliers(X)
        X_scaled = self._scale_features(X_processed, fit=True)
        
        # Compute class weights for imbalanced data
        self.model_params['class_weight'] = self._compute_class_weights(y)
        
        # Initialize and fit model
        self.model = self.model_class(**self.model_params)
        self.model.fit(X_scaled, y)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions."""
        X_processed = self._handle_outliers(X, reference=X)
        X_scaled = self._scale_features(X_processed)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class predictions."""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self._get_feature_importance()

class TimeSeriesEnsemble:
    """Ensemble model specifically designed for time series prediction."""
    
    def __init__(self, base_models: List[Tuple[str, Any, Dict[str, Any]]]):
        """Initialize with list of (name, model_class, params) tuples."""
        self.base_models = [
            TimeAwareModel(model_class, params)
            for _, model_class, params in base_models
        ]
        self.model_names = [name for name, _, _ in base_models]
        self.weights = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> None:
        """Fit all base models and compute optimal weights."""
        # Split data for weight optimization
        cutoff_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:cutoff_idx], X.iloc[cutoff_idx:]
        y_train, y_val = y.iloc[:cutoff_idx], y.iloc[cutoff_idx:]
        dates_train, dates_val = dates.iloc[:cutoff_idx], dates.iloc[cutoff_idx:]
        
        # Fit base models
        print("Training base models...")
        for model, name in zip(self.base_models, self.model_names):
            print(f"Training {name}...")
            model.fit(X_train, y_train, dates_train)
        
        # Get predictions for weight optimization
        predictions = np.array([
            model.predict_proba(X_val)
            for model in self.base_models
        ])
        
        # Optimize weights using log loss
        print("Optimizing ensemble weights...")
        self.weights = self._optimize_weights(predictions, y_val)
        print("Ensemble weights:", dict(zip(self.model_names, self.weights)))
    
    def _optimize_weights(self, predictions: np.ndarray, y_true: pd.Series) -> np.ndarray:
        """Optimize ensemble weights using log loss."""
        from scipy.optimize import minimize
        
        def weighted_log_loss(weights):
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            ensemble_pred = np.sum(
                [w * p for w, p in zip(weights, predictions)],
                axis=0
            )
            return log_loss(y_true, ensemble_pred)
        
        # Initial weights (equal weighting)
        initial_weights = np.ones(len(self.base_models)) / len(self.base_models)
        
        # Optimize with constraints (weights sum to 1 and are non-negative)
        bounds = [(0, 1) for _ in range(len(self.base_models))]
        constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        result = minimize(
            weighted_log_loss,
            initial_weights,
            bounds=bounds,
            constraints=constraint,
            method='SLSQP'
        )
        
        return result.x
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate weighted ensemble predictions."""
        predictions = np.array([
            model.predict_proba(X)
            for model in self.base_models
        ])
        
        return np.sum(
            [w * p for w, p in zip(self.weights, predictions)],
            axis=0
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class predictions."""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from all base models."""
        return {
            name: model.get_feature_importance()
            for name, model in zip(self.model_names, self.base_models)
        }

def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
    """Compute comprehensive evaluation metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_pred_proba),
        'precision': dict(zip(['H', 'D', 'A'], precision)),
        'recall': dict(zip(['H', 'D', 'A'], recall)),
        'f1': dict(zip(['H', 'D', 'A'], f1)),
        'support': dict(zip(['H', 'D', 'A'], support))
    }
