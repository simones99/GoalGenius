from pathlib import Path

from goalline.settings import Settings

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_matches.csv"

SMOKE = Settings(
    first_season=2000,
    train_first_season=2000,
    dev_seasons=(2002, 2003),
    final_seasons=(2004, 2005),
    elo_k=(20,),
    elo_h=(60,),
    elo_r=(0.0, 0.33),
    draw_peak=(0.26, 0.28),
    draw_scale=(400,),
    dc_xi=(0.0019,),
    logistic_c=(1.0,),
    xgb_max_depth=(2,),
    xgb_learning_rate=(0.1,),
    xgb_min_child_weight=(1,),
    bootstrap_replicates=200,
)
