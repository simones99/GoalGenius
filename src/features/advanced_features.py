import pandas as pd
import numpy as np
from typing import List

def add_league_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add league position and points features, ensuring no future data leakage."""
    df = df.copy()
    
    # Sort by division and date to ensure temporal order
    df = df.sort_values(['Division', 'MatchDate'])
    
    # Initialize points columns
    df['HomePoints'] = df['FTResult'].map({'H': 3, 'D': 1, 'A': 0})
    df['AwayPoints'] = df['FTResult'].map({'H': 0, 'D': 1, 'A': 3})
    
    # Calculate season points and position using only past matches
    for division in df['Division'].unique():
        div_mask = df['Division'] == division
        
        # Get unique seasons based on date
        df.loc[div_mask, 'Season'] = df.loc[div_mask, 'MatchDate'].dt.year + (
            df.loc[div_mask, 'MatchDate'].dt.month > 7).astype(int)
        
        for season in df.loc[div_mask, 'Season'].unique():
            season_mask = (df['Season'] == season) & div_mask
            
            # Previous season's final standings
            prev_season = df[
                (df['Season'] == season - 1) & 
                div_mask
            ].copy()
            
            if not prev_season.empty:
                # Calculate previous season's total points
                home_points = prev_season.groupby('HomeTeam')['HomePoints'].sum()
                away_points = prev_season.groupby('AwayTeam')['AwayPoints'].sum()
                total_points = pd.DataFrame({
                    'Points': home_points.add(away_points, fill_value=0)
                }).fillna(0)
                
                # Create position mapping
                positions = total_points.rank(ascending=False, method='min')
                
                # Map previous season position to current season teams
                for team_type in ['Home', 'Away']:                        df.loc[season_mask, f'{team_type}PrevSeasonPos'] = (
                        df.loc[season_mask, f'{team_type}Team']
                        .map(positions['Points'])
                        .fillna(total_points['Points'].max() + 1)  # New teams get worst position + 1
                    )
            
            # Calculate current form using only past matches in rolling window
            for team_type in ['Home', 'Away']:
                points_col = f'{team_type}Points'
                team_col = f'{team_type}Team'
                
                # Last 5 matches points (only using past matches)
                df.loc[season_mask, f'{team_type}Form5'] = df[season_mask].groupby(team_col)[points_col].transform(
                    lambda x: x.shift().rolling(window=5, min_periods=1).sum()
                )
                
                # Last 10 matches points
                df.loc[season_mask, f'{team_type}Form10'] = df[season_mask].groupby(team_col)[points_col].transform(
                    lambda x: x.shift().rolling(window=10, min_periods=1).sum()
                )
    
    # Calculate form differences
    df['Form5Diff'] = df['HomeForm5'] - df['AwayForm5']
    df['Form10Diff'] = df['HomeForm10'] - df['AwayForm10']
    df['PrevSeasonPosDiff'] = df['HomePrevSeasonPos'] - df['AwayPrevSeasonPos']
    
    return df

def add_advanced_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced form metrics including goals and match stats."""
    df = df.copy()
    
    # Goals scored/conceded rolling averages
    for team_type in ['Home', 'Away']:
        # Goals scored
        df[f'{team_type}GoalsScored'] = df.groupby(['Division', f'{team_type}Team'])[f'FT{team_type}'].transform(
            lambda x: x.shift().rolling(window=5, min_periods=1).mean()
        )
        
        # Goals conceded (opposite team's goals)
        opposite = 'Away' if team_type == 'Home' else 'Home'
        df[f'{team_type}GoalsConceded'] = df.groupby(['Division', f'{team_type}Team'])[f'FT{opposite}'].transform(
            lambda x: x.shift().rolling(window=5, min_periods=1).mean()
        )
        
    # Goal difference features
    df['GoalsScoredDiff'] = df['HomeGoalsScored'] - df['AwayGoalsScored']
    df['GoalsConcededDiff'] = df['HomeGoalsConceded'] - df['AwayGoalsConceded']
    
    return df

def add_match_stats_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling averages for match statistics with proper missing value handling and no data leakage."""
    df = df.copy()
    
    stats_columns = [
        'Shots', 'Target', 'Corners', 'Fouls',
        'Yellow', 'Red'
    ]
    
    windows = [3, 5, 10]  # Multiple time windows for better trend capture
    
    # Sort by date to ensure temporal order
    df = df.sort_values('MatchDate')
    
    for stat in stats_columns:
        for team_type in ['Home', 'Away']:
            col = f'{team_type}{stat}'
            team_col = f'{team_type}Team'
            
            # Fill missing values with expanding team mean (using only past data)
            df[col] = df.groupby(['Division', team_col])[col].transform(
                lambda x: x.fillna(x.expanding().mean())
            )
            
            # For each division separately to prevent division mixing
            for division in df['Division'].unique():
                div_mask = df['Division'] == division
                
                # Calculate rolling averages for different windows
                for window in windows:
                    df.loc[div_mask, f'{col}Avg{window}'] = df[div_mask].groupby(team_col)[col].transform(
                        lambda x: x.shift().rolling(window=window, min_periods=1).mean()
                    )
                    
                    # Add variance to capture consistency
                    df.loc[div_mask, f'{col}Var{window}'] = df[div_mask].groupby(team_col)[col].transform(
                        lambda x: x.shift().rolling(window=window, min_periods=2).var()
                    ).fillna(0)
    
    # Calculate differences and normalized differences
    for stat in stats_columns:
        for window in windows:
            # Basic difference in averages
            home_avg = df[f'Home{stat}Avg{window}']
            away_avg = df[f'Away{stat}Avg{window}']
            
            df[f'{stat}Diff{window}'] = home_avg - away_avg
            
            # Normalized difference (by total)
            total = home_avg + away_avg
            df[f'{stat}NormDiff{window}'] = (
                (home_avg - away_avg) / total.replace(0, 1)
            )
            
            # Consistency comparison
            home_var = df[f'Home{stat}Var{window}']
            away_var = df[f'Away{stat}Var{window}']
            df[f'{stat}ConsistencyDiff{window}'] = (
                (away_var - home_var) / (home_var + away_var).replace(0, 1)
            )
    
    # Create advanced composite features
    for window in windows:
        # Attacking threat (weighted combination of shots and corners)
        df[f'HomeAttackThreat{window}'] = (
            0.6 * df[f'HomeShotsAvg{window}'] + 
            0.3 * df[f'HomeTargetAvg{window}'] +
            0.1 * df[f'HomeCornersAvg{window}']
        ) / df[f'HomeShotsVar{window}'].replace(0, 1)  # Penalize inconsistency
        
        df[f'AwayAttackThreat{window}'] = (
            0.6 * df[f'AwayShotsAvg{window}'] + 
            0.3 * df[f'AwayTargetAvg{window}'] +
            0.1 * df[f'AwayCornersAvg{window}']
        ) / df[f'AwayShotsVar{window}'].replace(0, 1)  # Penalize inconsistency
        
        # Discipline score (weighted combination with consistency penalty)
        df[f'HomeDiscipline{window}'] = (
            0.5 * df[f'HomeFoulsAvg{window}'] + 
            0.35 * df[f'HomeYellowAvg{window}'] + 
            0.15 * df[f'HomeRedAvg{window}']
        ) * (1 + df[f'HomeFoulsVar{window}'])  # Higher variance = worse discipline
        
        df[f'AwayDiscipline{window}'] = (
            0.5 * df[f'AwayFoulsAvg{window}'] + 
            0.35 * df[f'AwayYellowAvg{window}'] + 
            0.15 * df[f'AwayRedAvg{window}']
        ) * (1 + df[f'AwayFoulsVar{window}'])  # Higher variance = worse discipline
        
        # Shot efficiency (conversion rate with consistency bonus)
        df[f'HomeEfficiency{window}'] = (
            df[f'HomeTargetAvg{window}'] / df[f'HomeShotsAvg{window}'].replace(0, 1)
        ) * (1 - df[f'HomeShotsVar{window}'].clip(0, 0.5))  # Bonus for consistency
        
        df[f'AwayEfficiency{window}'] = (
            df[f'AwayTargetAvg{window}'] / df[f'AwayShotsAvg{window}'].replace(0, 1)
        ) * (1 - df[f'AwayShotsVar{window}'].clip(0, 0.5))  # Bonus for consistency
        
    return df

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic features required for interactions."""
    df = df.copy()
    
    # Create basic form difference
    df['FormDiff'] = df['HomeForm5']  # This will be properly calculated in add_league_features
    
    # Add derby flag based on teams from same city/region
    df['IsDerby'] = False  # We'll need proper team location data to improve this
    
    # Add Elo difference if Elo columns exist
    if 'HomeElo' in df.columns and 'AwayElo' in df.columns:
        df['EloDiff'] = df['HomeElo'] - df['AwayElo']
    else:
        # Create placeholder Elo difference based on team strength indicators
        df['EloDiff'] = (
            df['HomeForm10'].fillna(0) - df['AwayForm10'].fillna(0) + 
            (df['HomePrevSeasonPos'].fillna(20) - df['AwayPrevSeasonPos'].fillna(20)) * -0.5
        )
    
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between key metrics."""
    df = df.copy()
    
    # Ensure required features exist
    df = add_basic_features(df)
    
    # Weight recent form more heavily than older form
    def weighted_form(form5, form10):
        return 0.7 * form5 + 0.3 * form10
    
    # Create intermediate weighted features
    weighted_features = {
        'WeightedHomeForm': weighted_form(df['HomeForm5'], df['HomeForm10']),
        'WeightedAwayForm': weighted_form(df['AwayForm5'], df['AwayForm10']),
        'WeightedFormDiff': weighted_form(df['Form5Diff'], df['Form10Diff']),
        
        # Weighted attacking threat (using multiple windows)
        'WeightedHomeAttack': weighted_form(df['HomeAttackThreat5'], df['HomeAttackThreat10']),
        'WeightedAwayAttack': weighted_form(df['AwayAttackThreat5'], df['AwayAttackThreat10']),
        'WeightedAttackDiff': weighted_form(
            df['HomeAttackThreat5'] - df['AwayAttackThreat5'],
            df['HomeAttackThreat10'] - df['AwayAttackThreat10']
        ),
        
        # Weighted defensive metrics
        'WeightedHomeDiscipline': weighted_form(df['HomeDiscipline5'], df['HomeDiscipline10']),
        'WeightedAwayDiscipline': weighted_form(df['AwayDiscipline5'], df['AwayDiscipline10']),
        'WeightedDisciplineDiff': weighted_form(
            df['HomeDiscipline5'] - df['AwayDiscipline5'],
            df['HomeDiscipline10'] - df['AwayDiscipline10']
        ),
        
        # Weighted efficiency
        'WeightedHomeEfficiency': weighted_form(df['HomeEfficiency5'], df['HomeEfficiency10']),
        'WeightedAwayEfficiency': weighted_form(df['AwayEfficiency5'], df['AwayEfficiency10']),
        'WeightedEfficiencyDiff': weighted_form(
            df['HomeEfficiency5'] - df['AwayEfficiency5'],
            df['HomeEfficiency10'] - df['AwayEfficiency10']
        )
    }
    
    # Add weighted features
    df = df.assign(**weighted_features)
    
    # Create enhanced interactions
    new_features = {
        # Basic interactions
        'EloFormInteraction': df['EloDiff'] * df['WeightedFormDiff'],
        'EloAttackInteraction': df['EloDiff'] * df['WeightedAttackDiff'],
        'EloDisciplineInteraction': df['EloDiff'] * df['WeightedDisciplineDiff'],
        'EloEfficiencyInteraction': df['EloDiff'] * df['WeightedEfficiencyDiff'],
        
        # Form interactions
        'FormAttackInteraction': df['WeightedFormDiff'] * df['WeightedAttackDiff'],
        'FormDisciplineInteraction': df['WeightedFormDiff'] * df['WeightedDisciplineDiff'],
        'FormEfficiencyInteraction': df['WeightedFormDiff'] * df['WeightedEfficiencyDiff'],
        
        # Match stats interactions
        'AttackDefenseInteraction': df['WeightedAttackDiff'] * df['WeightedDisciplineDiff'],
        'AttackEfficiencyInteraction': df['WeightedAttackDiff'] * df['WeightedEfficiencyDiff'],
        
        # Derby interactions
        'DerbyFormInteraction': df['IsDerby'] * df['WeightedFormDiff'],
        'DerbyEloInteraction': df['IsDerby'] * df['EloDiff'],
        'DerbyAttackInteraction': df['IsDerby'] * df['WeightedAttackDiff'],
        
        # Season position impact
        'SeasonPositionFormInteraction': df['PrevSeasonPosDiff'] * df['WeightedFormDiff'],
        'SeasonPositionEloInteraction': df['PrevSeasonPosDiff'] * df['EloDiff'],
        
        # Combined strength indicators
        'OverallStrengthHome': (
            df['WeightedHomeForm'] * 0.3 +
            df['WeightedHomeAttack'] * 0.3 +
            df['WeightedHomeEfficiency'] * 0.2 +
            df['HomePrevSeasonPos'].clip(1, 20).map(lambda x: 1/x) * 0.2
        ),
        'OverallStrengthAway': (
            df['WeightedAwayForm'] * 0.3 +
            df['WeightedAwayAttack'] * 0.3 +
            df['WeightedAwayEfficiency'] * 0.2 +
            df['AwayPrevSeasonPos'].clip(1, 20).map(lambda x: 1/x) * 0.2
        )
    }
    
    # Add composite strength difference
    new_features['OverallStrengthDiff'] = new_features['OverallStrengthHome'] - new_features['OverallStrengthAway']
    
    return df.assign(**new_features)
