from scipy.stats import randint, uniform

PARAM_DISTRIBUTIONS = {
    'random_forest': {
        'n_estimators': randint(500, 1500),
        'max_depth': randint(15, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced_subsample'],  # Removed duplicate 'balanced' option
        'bootstrap': [True],
        'max_samples': uniform(0.5, 0.5),
        'ccp_alpha': uniform(0, 0.01)
    },
    'xgboost': {
        'n_estimators': randint(500, 1500),
        'max_depth': randint(6, 15),
        'learning_rate': uniform(0.001, 0.1),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'min_child_weight': randint(1, 7),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1.0),
        'reg_lambda': uniform(1, 4.0),
        'max_delta_step': randint(1, 10)
    }
}

MODEL_CONFIGS = {
    'logistic': {
        'class': 'LogisticRegression',
        'params': {
            'multi_class': 'multinomial',
            'max_iter': 10000,
            'solver': 'lbfgs',
            'random_state': 42,
            'class_weight': 'balanced',
            'C': 0.1,
            'penalty': 'l2'
        }
    },
    'random_forest': {
        'class': 'RandomForestClassifier',
        'params': {
            'random_state': 42,
            'verbose': 0,
            'n_jobs': -1,
            'criterion': 'entropy',
            'class_weight': 'balanced_subsample',  # Changed to match hyperparameter search
            'warm_start': True,
            'n_estimators': 1000,  # Sensible default
            'max_depth': 30
        }
    },
    'xgboost': {
        'class': 'XGBClassifier',
        'params': {
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1,
            'tree_method': 'hist',
            'eval_metric': ['mlogloss', 'auc_mu'],
            'early_stopping_rounds': 50,
            'learning_rate': 0.01,
            'max_depth': 10,
            'n_estimators': 1000,
            'scale_pos_weight': None  # Will be calculated dynamically
        }
    }
}