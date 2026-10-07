# GoalGenius

GoalGenius is a machine learning project that predicts the outcome of club football matches (home win, draw, away win) from historical match data and Elo ratings (2000–2025). It uses Python, pandas, scikit-learn and XGBoost, with time-based splits to avoid look-ahead leakage.

## Project Goal
- **Objective:** Produce well-calibrated, low log-loss probabilities for football match outcomes from historical CSV data.
- **Status:** Offline training and evaluation pipeline. A prediction API and dashboard are planned (see Next Steps) and not implemented yet.

## Achievements So Far
- **Data Pipeline:**
  - Ingested and cleaned historical match data (2000–2025) from CSVs.
  - Engineered features including Elo difference, recent form, head-to-head, and derby indicators.
- **Modeling:**
  - Enhanced model training pipeline with proper class imbalance handling
  - Implemented improved stacking ensemble with time-series cross-validation
  - Added early stopping for XGBoost to prevent overfitting
  - Optimized Random Forest with balanced subsample weights
  - Standardized feature scaling across all models
  - Used time-based train/validation/test splits to avoid data leakage
  - Performed hyperparameter tuning with RandomizedSearchCV for tree-based models
  - Exported best model and metrics for reproducibility
- **MLOps:**
  - Organized code into modular structure (src/, models/, data/, notebooks/)
  - Automated model training and evaluation pipeline
  - Improved error handling and logging
  - Set up .gitignore and requirements.txt for clean version control and reproducibility

## Results

Validation metrics from `python -m models.train` (saved in `data/results/metrics_ensemble_20250531_131355.json`). The split is by date: 45,073 training matches and 691 validation matches (Feb–Aug 2024). Baselines are computed on the same validation period, using class frequencies from the training period.

| Model | Accuracy | Log loss |
|---|---|---|
| Logistic regression (Elo diff, form diff, head-to-head, derby) | 0.498 | 1.029 |
| Majority class (always home win) | 0.412 | — |
| Training-period class frequencies | — | 1.087 |
| Uniform guess | 0.333 | 1.099 |

The model beats the naive baselines, but three-way football outcomes are hard to predict and bookmaker-implied probabilities are the usual benchmark. The validation set was also used for model selection, so these numbers are slightly optimistic. The logistic regression result reproduces exactly with the pinned requirements.

### Known issues
- The XGBoost step fails with XGBoost 3.x because `auc_mu` is not a valid `eval_metric`.
- The stacking ensemble fails because `cross_val_predict` does not accept `TimeSeriesSplit`.

## How to run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Place Matches.csv and EloRatings.csv in data/raw/
python -m models.train            # baseline models (logistic, random forest, XGBoost)
python -m models.train_ensemble   # stacking ensemble
```

Metrics are written to `data/results/`. Trained models are saved to `data/models/` and are not tracked in git.

## Next Steps
- **Feature Engineering:**
  - Integrate additional features such as league position, points, and advanced form metrics
  - Explore external data sources (injuries, weather, odds movement) for further improvements
- **Modeling:**
  - Add model calibration techniques for better probability estimates
  - Implement Bayesian optimization for hyperparameter tuning
  - Analyze feature importance and model explainability
  - Consider adding LightGBM as another base model
- **Real-Time Integration:**
  - Connect to API-Football for live fixtures and stats
  - Implement caching and feature transformation for real-time predictions
- **Deployment:**
  - Serve predictions via FastAPI endpoint (`GET /predict?fixture_id=<id>`)
  - Build a Streamlit dashboard for next week's matches and model insights
  - Containerize the app with Docker for easy deployment
- **Automation:**
  - Set up a cron job to refresh the API cache and update predictions daily
  - Implement model retraining pipeline with performance monitoring

---

**Repository Structure:**
- `src/` — Feature engineering and data ingestion code
- `models/` — Model training, evaluation, and utilities
- `data/` — Raw, processed, and results data
  - `models/` — Saved model files (not tracked)
  - `results/` — Training metrics and evaluation results
- `notebooks/` — EDA and prototyping
- `requirements.txt` — Project dependencies

For more details, see the code and documentation in each folder.