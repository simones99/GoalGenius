import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .advanced_features import (
    add_league_features,
    add_advanced_form_features,
    add_match_stats_features,
    create_interaction_features
)

FEATURE_DESCRIPTIONS = {
    'EloDiff': 'Difference in Elo ratings (Home - Away)',
    'FormDiff': 'Difference in recent form (Home - Away)',
    'H2H': 'Head-to-head historical performance score',
    'IsDerby': 'Boolean indicating if match is a derby (1) or not (0)',
}

def handle_outliers_iqr(data: pd.DataFrame, 
                       reference_data: pd.DataFrame = None,
                       method: str = 'clip') -> pd.DataFrame:
    """
    Handle outliers using IQR method for numeric columns only.
    
    Args:
        data: Input DataFrame
        reference_data: Optional reference DataFrame to compute IQR bounds (e.g., training data)
        method: Either 'clip' to cap at IQR boundaries or 'remove' to filter
    
    Returns:
        DataFrame with outliers handled
    """
    # Make a copy of the data
    data_processed = data.copy()
    
    # Get numeric columns only
    numeric_cols = data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    
    if not numeric_cols.empty:
        # Use reference data if provided, otherwise use input data
        ref = reference_data if reference_data is not None else data
        
        Q1 = ref[numeric_cols].quantile(0.25)
        Q3 = ref[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == 'clip':
            data_processed[numeric_cols] = data[numeric_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)
        else:  # method == 'remove'
            outlier_mask = ((data[numeric_cols] < lower_bound) | (data[numeric_cols] > upper_bound)).any(axis=1)
            data_processed = data_processed[~outlier_mask]
    
    return data_processed

def compute_h2h_score(df: pd.DataFrame, decay_factor: float = 0.5) -> np.ndarray:
    """
    Create H2H feature using all available results with exponential decay.
    More recent matches have higher impact on the H2H score.

    Args:
        df: DataFrame with 'HomeTeam', 'AwayTeam', 'FTResult', 'MatchDate'
        decay_factor: Decay factor for weighting past matches (0 < decay_factor < 1)

    Returns:
        numpy.ndarray: Array of H2H scores for each match in the DataFrame
    """
    # Pre-sort DataFrame by date
    df_sorted = df.sort_values("MatchDate")
    n_matches = len(df_sorted)
    
    # Pre-allocate array for scores
    h2h_scores = np.zeros(n_matches)
    h2h_history = {}
    
    # Convert to numpy arrays for faster access
    home_teams = df_sorted['HomeTeam'].values
    away_teams = df_sorted['AwayTeam'].values
    results = df_sorted['FTResult'].values
    
    for i in range(n_matches):
        home = home_teams[i]
        away = away_teams[i]
        pair = tuple(sorted([home, away]))
        history = h2h_history.get(pair, [])
        
        # Calculate exponentially decaying weighted score
        if history:
            score = 0.0
            total_weight = 0.0
            for j, (team, result_value) in enumerate(reversed(history)):
                weight = decay_factor ** j
                total_weight += weight
                if team == home:
                    score += result_value * weight
                else:
                    score -= result_value * weight
            h2h_scores[i] = score / total_weight if total_weight > 0 else 0.0
        
        # Update history with current match result
        result_value = 3 if results[i] == 1 else (-3 if results[i] == 2 else 0)
        history.append((home, result_value))
        h2h_history[pair] = history
    
    return h2h_scores

def add_derby_feature(df):
    """Add derby feature based on city/regional rivalries."""
    # Define major derbies and rivalries for each league
    
    derbies = {
    'E1': [  # Premier League
        {'Manchester': ['Man City', 'Man United', 'Manchester City', 'Manchester Utd']},
        {'London': ['Arsenal', 'Chelsea', 'Tottenham', 'West Ham', 'Crystal Palace', 'Fulham', 
                   'Brentford', 'QPR', 'Charlton', 'Millwall']},
        {'Liverpool': ['Liverpool', 'Everton']},
        {'North East': ['Newcastle', 'Sunderland', 'Middlesbrough']},
        {'Midlands': ['Aston Villa', 'Birmingham', 'West Brom', 'Wolves', 'Leicester', 
                     'Nott\'m Forest', 'Derby', 'Stoke']},
    ],
    'I1': [  # Serie A
        {'Milan': ['Milan', 'Inter', 'AC Milan', 'Internazionale']},
        {'Rome': ['Roma', 'Lazio', 'AS Roma', 'SS Lazio']},
        {'Turin': ['Juventus', 'Torino']},
        {'Genoa': ['Genoa', 'Sampdoria']},
        {'Verona': ['Verona', 'Chievo', 'Hellas Verona']},
        {'Tuscany': ['Fiorentina', 'Empoli', 'Siena']},
        {'Sicily': ['Palermo', 'Catania']},
    ],
    'SP1': [  # La Liga
        {'Madrid': ['Real Madrid', 'Atletico Madrid', 'Getafe', 'Rayo Vallecano', 'Leganes']},
        {'Barcelona': ['Barcelona', 'Espanyol', 'Girona']},
        {'Seville': ['Sevilla', 'Betis', 'Real Betis']},
        {'Basque': ['Ath Bilbao', 'Athletic Bilbao', 'Real Sociedad', 'Alaves', 'Eibar', 
                    'Osasuna']},
        {'Valencia': ['Valencia', 'Levante', 'Villarreal', 'Elche']},
    ],
    'D1': [  # Bundesliga
        {'Berlin': ['Hertha', 'Union Berlin']},
        {'Ruhr': ['Schalke 04', 'Dortmund', 'Borussia Dortmund', 'Bochum']},
        {'Bavaria': ['Bayern Munich', 'Augsburg', 'Nurnberg']},
        {'Hamburg': ['Hamburg', 'St Pauli']},
        {'Rhine': ['FC Koln', 'Cologne', 'Leverkusen', 'Bayer Leverkusen', 
                  'M\'gladbach', 'Fortuna Dusseldorf', 'Mainz']},
    ],
    'F1': [  # Ligue 1
        {'Paris': ['Paris SG', 'PSG', 'Paris FC']},
        {'Rhone-Alps': ['Lyon', 'St Etienne', 'Saint-Etienne']},
        {'Cote dAzur': ['Nice', 'Monaco', 'Marseille']},
        {'North': ['Lille', 'Lens', 'Valenciennes']},
        {'Brittany': ['Rennes', 'Nantes', 'Brest', 'Lorient']},
        {'Normandy': ['Caen', 'Le Havre']},
    ]
}
    
    def is_derby(row):
        if row['Division'] not in derbies:
            return 0
        
        for city_derbies in derbies[row['Division']]:
            for city, teams in city_derbies.items():
                if row['HomeTeam'] in teams and row['AwayTeam'] in teams:
                    return 1
        return 0
    
    df['IsDerby'] = df.apply(is_derby, axis=1)
    return df



def build_features(df: pd.DataFrame, 
                   drop_intermediate: bool = True) -> pd.DataFrame:
    """
    Apply core feature transformations to raw match data.

    Parameters:
    - df (pd.DataFrame): Raw match data.
    - drop_intermediate (bool): If True, drop columns used only for intermediate calculations.

    Returns:
    - pd.DataFrame: Feature-engineered DataFrame ready for modeling.
    """
    # Add input validation
    required_columns = [
        'HomeElo', 'AwayElo', 
        'Form3Home', 'Form5Home', 
        'Form3Away', 'Form5Away',
        'HomeTeam', 'AwayTeam',
        'MatchDate', 'Division',
        'FTResult', 'FTHome', 'FTAway',
        'HomeShots', 'AwayShots',
        'HomeTarget', 'AwayTarget',
        'HomeFouls', 'AwayFouls',
        'HomeCorners', 'AwayCorners',
        'HomeYellow', 'AwayYellow',
        'HomeRed', 'AwayRed'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate basic features
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]
    df["HomeFormScore"] = 0.6 * df["Form3Home"] + 0.4 * df["Form5Home"]
    df["AwayFormScore"] = 0.6 * df["Form3Away"] + 0.4 * df["Form5Away"]
    df["FormDiff"] = df["HomeFormScore"] - df["AwayFormScore"]
    df["H2H"] = compute_h2h_score(df)
    df = add_derby_feature(df)
    
    # Add advanced features
    df = add_league_features(df)
    df = add_advanced_form_features(df)
    df = add_match_stats_features(df)
    df = create_interaction_features(df)
    
    if drop_intermediate:
        intermediate_cols = [
            "HomeElo", "AwayElo",
            "Form3Home", "Form5Home", 
            "Form3Away", "Form5Away",
            "HomeFormScore", "AwayFormScore",
            "HomePoints", "AwayPoints"
        ]
        df.drop(columns=intermediate_cols, inplace=True, errors="ignore")
          
    return df

def build_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build head-to-head features for each match."""
    df = df.copy()
    
    # Sort by date to ensure temporal ordering
    df = df.sort_values('MatchDate')
    
    # Compute H2H score using exponential decay
    df['H2H'] = compute_h2h_score(df, decay_factor=0.5)
    
    # Add H2H goals stats (last 5 meetings)
    df = df.sort_values('MatchDate')
    h2h_pairs = {}
    
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        pair = tuple(sorted([home, away]))
        
        # Get previous meetings
        if pair not in h2h_pairs:
            h2h_pairs[pair] = []
            df.at[idx, 'H2HGoalsDiff5'] = 0
            df.at[idx, 'H2HGoalsTotal5'] = 0
        else:
            meetings = h2h_pairs[pair][-5:]  # Last 5 meetings
            if meetings:
                goals_diff = sum([m[1] if m[0] == home else -m[1] for m in meetings]) / len(meetings)
                goals_total = sum([abs(m[1]) for m in meetings]) / len(meetings)
                df.at[idx, 'H2HGoalsDiff5'] = goals_diff
                df.at[idx, 'H2HGoalsTotal5'] = goals_total
            else:
                df.at[idx, 'H2HGoalsDiff5'] = 0
                df.at[idx, 'H2HGoalsTotal5'] = 0
        
        # Update H2H history with current match
        home_goals = 1 if row['FTResult'] == 'H' else (-1 if row['FTResult'] == 'A' else 0)
        h2h_pairs[pair].append((home, home_goals))
    
    # Fill any missing values
    df['H2HGoalsDiff5'] = df['H2HGoalsDiff5'].fillna(0)
    df['H2HGoalsTotal5'] = df['H2HGoalsTotal5'].fillna(0)
    
    return df