import pandas as pd

# Add at the top of the file
FEATURE_DESCRIPTIONS = {
    'EloDiff': 'Difference in Elo ratings (Home - Away)',
    'FormDiff': 'Difference in recent form (Home - Away)',
    'H2H': 'Head-to-head historical performance score',
    'IsDerby': 'Boolean indicating if match is a derby (1) or not (0)',
}

def handle_outliers_iqr(data: pd.DataFrame, 
                       method: str = 'clip') -> pd.DataFrame:
    """
    Handle outliers using IQR method.
    
    Args:
        data: Input DataFrame
        method: Either 'clip' to cap at IQR boundaries or 'remove' to filter
    
    Returns:
        DataFrame with outliers handled
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    if method == 'clip':
        return data.clip(lower=lower_bound, upper=upper_bound, axis=1)
    else:  # method == 'remove'
        outlier_mask = ((data < lower_bound) | (data > upper_bound)).any(axis=1)
        return data[~outlier_mask]

def compute_h2h_score(df, decay_factor=0.5):
    """
    Create H2H feature using all available results with exponential decay.
    More recent matches have higher impact on the H2H score.

    Args:
        df (DataFrame): DataFrame with 'HomeTeam', 'AwayTeam', 'FTResult', 'MatchDate'.
        decay_factor (float): Decay factor for weighting past matches (0 < decay_factor < 1).

    Returns:
        List: List of H2H scores for each match in the DataFrame.
    """
    df_sorted = df.sort_values("MatchDate")
    h2h_scores = []
    h2h_history = {}

    for idx, row in df_sorted.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        pair = tuple(sorted([home, away]))
        history = h2h_history.get(pair, [])

        # Calculate exponentially decaying weighted score from history (from home team's perspective)
        if not history:
            h2h_scores.append(0)
        else:
            score = 0
            total_weight = 0
            # Most recent match is last in history
            for i, match in enumerate(reversed(history)):
                weight = decay_factor ** i
                total_weight += weight
                if match[0] == home:
                    score += match[1] * weight
                else:
                    score -= match[1] * weight  # invert if the other team was home
            h2h_scores.append(score / total_weight if total_weight > 0 else 0)

        # Update history: always store (team, result_value) where team is the home team in that match
        result_value = 3 if row["FTResult"] == 1 else (-3 if row["FTResult"] == 2 else 0)
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
                   drop_intermediate: bool = True, 
                   drop_unused: bool = True) -> pd.DataFrame:
    """
    Apply core feature transformations to raw match data.

    Parameters:
    - df (pd.DataFrame): Raw match data.
    - drop_intermediate (bool): If True, drop columns used only for intermediate calculations.
    - drop_unused (bool): If True, drop unused match stats like shots, fouls, cards.

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
        'FTResult'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate Elo difference
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]

    # Calculate weighted form scores
    df["HomeFormScore"] = 0.6 * df["Form3Home"] + 0.4 * df["Form5Home"]
    df["AwayFormScore"] = 0.6 * df["Form3Away"] + 0.4 * df["Form5Away"]

    # Calculate Form difference
    df["FormDiff"] = df["HomeFormScore"] - df["AwayFormScore"]
    
    # Calculate H2H score
    df["H2H"] = compute_h2h_score(df)
    
    # Add derby feature
    df = add_derby_feature(df)

    if drop_intermediate:
        # Drop intermediate columns if not needed
        df.drop(columns=["HomeElo", 
                         "AwayElo",
                         "Form3Home", 
                         "Form5Home", 
                         "Form3Away",
                         "Form5Away", 
                         "HomeFormScore", 
                         "AwayFormScore",
                         "FTHome",
                         "FTAway",], 
                inplace=True,
                errors="ignore")

    if drop_unused:
        # Drop unused match stats
        df.drop(columns=[
        'HomeShots', 
        'AwayShots', 
        'HomeTarget', 
        'AwayTarget',
        'HomeFouls', 
        'AwayFouls',
        'HomeCorners', 
        'AwayCorners',
        'HomeYellow', 
        'AwayYellow', 
        'HomeRed', 
        'AwayRed'],
        inplace=True, 
        errors="ignore")
          
    return df