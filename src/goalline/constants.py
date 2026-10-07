"""Names shared by every module."""

FIRST_DIVISIONS = ("I1", "E0", "SP1", "D1", "F1")
SECOND_DIVISIONS = ("I2", "E1", "SP2", "D2", "F2")
DIVISIONS = FIRST_DIVISIONS + SECOND_DIVISIONS

COUNTRY = {
    "I1": "ITA", "I2": "ITA",
    "E0": "ENG", "E1": "ENG",
    "SP1": "ESP", "SP2": "ESP",
    "D1": "GER", "D2": "GER",
    "F1": "FRA", "F2": "FRA",
}

LEAGUE_NAMES = {
    "I1": "Serie A",
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

OUTCOMES = ("H", "D", "A")
OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}

ODDS_COLUMNS = {
    "b365": ("b365_h", "b365_d", "b365_a"),
    "max": ("max_h", "max_d", "max_a"),
}

MODEL_NAMES = ("uniform", "frequency", "elo", "dixon_coles", "logistic", "xgboost")
MAIN_MODELS = ("elo", "dixon_coles", "logistic", "xgboost")
REFERENCES = ("shin", "proportional", "max_odds")

MODEL_LABELS = {
    "uniform": "Uniform",
    "frequency": "Frequency",
    "elo": "Elo",
    "dixon_coles": "Dixon–Coles",
    "logistic": "Logistic regression",
    "xgboost": "XGBoost",
    "shin": "Bet365, Shin",
    "proportional": "Bet365, proportional",
    "max_odds": "Best odds, proportional",
}

SEED = 42
