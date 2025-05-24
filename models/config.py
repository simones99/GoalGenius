from scipy.stats import randint, uniform

PARAM_DISTRIBUTIONS = {
    'random_forest': {
        'n_estimators': randint(100, 500),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    },
    'xgboost': {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 7),
        'gamma': uniform(0, 0.5)
    }
}

MODEL_CONFIGS = {
    'logistic': {
        'class': 'LogisticRegression',
        'params': {
            'multi_class': 'multinomial',
            'max_iter': 2000,
            'solver': 'lbfgs',
            'random_state': 42
        }
    },
    'random_forest': {
        'class': 'RandomForestClassifier',
        'params': {
            'random_state': 42,
            'verbose': 0
        }
    },
    'xgboost': {
        'class': 'XGBClassifier',
        'params': {
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42,
            'verbosity': 0
        }
    }
}