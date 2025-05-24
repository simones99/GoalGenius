# GoalGenius

GoalGenius is a machine learning project designed to predict the outcome of club football matches (home win, draw, away win) using historical data and real-time features. The project leverages Python, pandas, scikit-learn, XGBoost, and modern MLOps practices to build, evaluate, and serve predictive models.

## Project Goal
- **Objective:** Achieve high-accuracy, low log-loss predictions for football match outcomes using a combination of historical CSV data and real-time API-Football data.
- **Use Case:** Provide real-time match forecasts via a FastAPI endpoint and a Streamlit dashboard for demo and analysis.

## Achievements So Far
- **Data Pipeline:**
  - Ingested and cleaned historical match data (2000–2025) from CSVs.
  - Engineered features including Elo difference, recent form, head-to-head, and derby indicators.
- **Modeling:**
  - Implemented baseline Logistic Regression, Random Forest, and XGBoost models.
  - Used time-based train/validation/test splits to avoid data leakage.
  - Performed hyperparameter tuning with RandomizedSearchCV for tree-based models.
  - Exported best model and metrics for reproducibility.
- **MLOps:**
  - Organized code into modular structure (src/, models/, data/, notebooks/).
  - Automated model training and evaluation pipeline.
  - Set up .gitignore and requirements.txt for clean version control and reproducibility.

## Next Steps
- **Feature Engineering:**
  - Integrate additional features such as league position, points, and advanced form metrics.
  - Explore external data sources (injuries, weather, odds movement) for further improvements.
- **Modeling:**
  - Experiment with ensemble models and advanced calibration techniques.
  - Analyze feature importance and model explainability.
- **Real-Time Integration:**
  - Connect to API-Football for live fixtures and stats.
  - Implement caching and feature transformation for real-time predictions.
- **Deployment:**
  - Serve predictions via FastAPI endpoint (`GET /predict?fixture_id=<id>`).
  - Build a Streamlit dashboard for next week’s matches and model insights.
  - Containerize the app with Docker for easy deployment.
- **Automation:**
  - Set up a cron job to refresh the API cache and update predictions daily.

---

**Repository Structure:**
- `src/` — Feature engineering and data ingestion code
- `models/` — Model training, evaluation, and utilities
- `data/` — Raw, processed, and results data
- `notebooks/` — EDA and prototyping
- `requirements.txt` — Project dependencies

For more details, see the code and documentation in each folder.