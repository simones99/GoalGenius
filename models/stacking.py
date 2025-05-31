from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from typing import List, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import xgboost as xgb

class StackingTimeSeriesClassifier(BaseEstimator, ClassifierMixin):
    """Custom stacking classifier for time series data."""
    
    def __init__(self, base_models: List[Any], meta_model: Any):
        self.base_models = base_models
        self.meta_model = meta_model
        self.fitted_base_models = []
        
    def fit(self, X, y):
        # First level: Train base models
        self.fitted_base_models = []
        
        # Use TimeSeriesSplit for proper validation
        tscv = TimeSeriesSplit(n_splits=5)
        meta_features = np.zeros((X.shape[0], len(self.base_models) * 3))
        
        # Train each base model and create meta-features
        for i, model in enumerate(self.base_models):
            print(f"Training base model {i+1}/{len(self.base_models)}...")
            
            # Fit on full training data
            fitted_model = model.fit(X, y)
            self.fitted_base_models.append(fitted_model)
            
            # Generate meta-features using cross-validation predictions
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                
                # Fit on training fold and predict on validation fold
                model.fit(X_train, y_train)
                fold_pred = model.predict_proba(X_val)
                meta_features[val_idx, i*3:(i+1)*3] = fold_pred
        
        # Second level: Train meta-model
        print("Training meta-model...")
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict_proba(self, X):
        # Generate predictions from base models
        meta_features = np.zeros((X.shape[0], len(self.base_models) * 3))
        
        for i, model in enumerate(self.fitted_base_models):
            pred = model.predict_proba(X)
            meta_features[:, i*3:(i+1)*3] = pred
        
        # Get final predictions from meta-model
        return self.meta_model.predict_proba(meta_features)
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
