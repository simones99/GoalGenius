"""Season ranges and hyperparameter grids. Defaults are the spec; tests use smaller ones."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    first_season: int = 2000
    train_first_season: int = 2002
    dev_seasons: tuple[int, ...] = tuple(range(2005, 2021))
    final_seasons: tuple[int, ...] = tuple(range(2021, 2026))
    elo_k: tuple[float, ...] = (10, 15, 20, 25, 30, 40)
    elo_h: tuple[float, ...] = (40, 60, 80, 100)
    elo_r: tuple[float, ...] = (0.0, 0.1, 0.2, 0.33, 0.5)
    draw_peak: tuple[float, ...] = (0.22, 0.24, 0.26, 0.28, 0.30, 0.32)
    draw_scale: tuple[float, ...] = (200, 300, 400, 500, 600)
    dc_xi: tuple[float, ...] = (0.0005, 0.001, 0.0019, 0.003)
    logistic_c: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    xgb_max_depth: tuple[int, ...] = (2, 3, 4)
    xgb_learning_rate: tuple[float, ...] = (0.03, 0.1)
    xgb_min_child_weight: tuple[float, ...] = (1, 10)
    bootstrap_replicates: int = 2000

    @property
    def final_first_season(self) -> int:
        return self.final_seasons[0]

    @property
    def last_season(self) -> int:
        return self.final_seasons[-1]
