from goalline.models.logistic import LogisticModel


def test_higher_elo_gap_raises_home_win_probability(features):
    model = LogisticModel(c=1.0, train_first_season=2000).fit(features[features.season < 2004])
    row = features[(features.season == 2004) & (features.tier == 1)].iloc[[0, 0]].copy()
    row["elo_diff"] = [-400.0, 400.0]
    p = model.predict_proba(row)
    assert p[1, 0] > p[0, 0]
    assert p[1, 2] < p[0, 2]


def test_trains_only_on_first_division_rows_from_the_first_season(features):
    model = LogisticModel(c=1.0, train_first_season=2001).fit(features[features.season < 2004])
    n_expected = len(features[(features.season.between(2001, 2003)) & (features.tier == 1)])
    assert model.n_train_ == n_expected
