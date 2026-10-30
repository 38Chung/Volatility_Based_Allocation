"""階段 3 的檢查：測試期日曆、模型訓練資料時點、凍結參數、前視偏誤、權重限制、繳交用 CSV。"""
import os

import numpy as np
import pandas as pd
import pytest

import config as C
import data_loader as dl
import run_stage2 as R2
import run_stage3 as R3


@pytest.fixture(scope="module")
def stage3():
    panel, sti = dl.load_panel()
    frozen = R3.load_frozen()
    ctx, view = R3.build_all(panel, sti, frozen)
    specs = R3.frozen_specs(frozen["params"])
    test = R3.run_groups(view, {g: specs[g] for g in ["A2", "S3", "B1"]}, ctx.days[-1])
    return dict(panel=panel, sti=sti, frozen=frozen, ctx=ctx, view=view, test=test)


def test_test_schedule(stage3):
    ctx, view = stage3["ctx"], stage3["view"]
    days = ctx.days
    first = days[days >= pd.Timestamp(C.TEST_START)][0]
    assert view.rebal[0] == first == pd.Timestamp("2026-10-01")            # 10/1（週四）初始建倉
    assert view.signal[0] == pd.Timestamp("2026-09-30")
    assert (view.signal < view.rebal).all()
    assert (days[days.get_indexer(view.rebal) - 1] == view.signal).all()
    for reb in view.rebal[1:]:                                              # 之後都是週二起第一個交易日
        tue = reb - pd.Timedelta(days=reb.day_of_week - C.REBALANCE_WEEKDAY)
        assert reb.day_of_week >= C.REBALANCE_WEEKDAY and days[(days >= tue) & (days <= reb)][0] == reb
    assert view.rebal[-1] <= min(days[-1], pd.Timestamp(C.TEST_END))
    weekly = pd.DatetimeIndex(ctx.sched["rebal"])
    expected = weekly[(weekly > first) & (weekly <= min(days[-1], pd.Timestamp(C.TEST_END)))]
    assert list(view.rebal[1:]) == list(expected)


def test_test_universe_is_members(stage3):
    """測試期只在當期成份股中選股；所有權重都在股票池內。"""
    view, test = stage3["view"], stage3["test"]
    m = dl.membership_mask(view.signal, "in_index").values
    assert not (view.tradable & ~m).any()
    for g, r in test.items():
        if g == "B0":
            continue
        W = r["weights"].values
        assert not ((W > 0) & ~m[:len(W)]).any(), g


def test_test_model_uses_finished_periods_only(stage3):
    log = stage3["view"].logs[stage3["frozen"]["params"]["LASSO_MODEL"]]
    assert len(log) == len(stage3["view"].signal)
    assert (pd.to_datetime(log["last_y_end"]) <= pd.to_datetime(log["signal"])).all()
    # 9/30 的模型最後一筆 y 是 9/22 → 9/29 那一期（9/29 → 10/6 那期還沒結束）
    assert pd.Timestamp(log["last_y_end"].iloc[0]) == pd.Timestamp("2026-09-29")


def test_frozen_params_used(stage3):
    p = stage3["frozen"]["params"]
    s = R3.frozen_specs(p)["S3"]
    assert s["K"] == p["K"] and s["weighting"] == p["WEIGHTING"] and s["sigma"] == p["SIGMA_METHOD"]
    assert s["model"] == p["LASSO_MODEL"] and s["exposure"]["target"] == p["EXPOSURE"]["target"]
    assert s["cost_bps"] == p["COST_BPS"] and s["sector_over"] == p["SECTOR_OVER"]
    W = stage3["test"]["S3"]["weights"]
    assert ((W > 0).sum(axis=1) <= p["K"]).all()


def test_test_weights(stage3):
    for g in ["A2", "S3", "B1"]:
        W, rb = stage3["test"][g]["weights"], stage3["test"][g]["rebal"]
        assert (W.values >= -1e-12).all()
        assert np.allclose(W.sum(axis=1).values, rb["exposure_target"].values, atol=1e-8)
        assert W.values.max() <= C.MAX_WEIGHT + 1e-8
    d = stage3["test"]["S3"]["daily"]
    assert d.index[0] == pd.Timestamp("2026-10-01")                         # 從 10/1 開盤建倉開始
    assert stage3["test"]["S3"]["rebal"]["cost"].iloc[0] > 0                 # 初始建倉要扣成本


def test_test_lookahead(stage3):
    """測試期內：把 10/13 之後的資料打亂，10/13（含）以前的分數、權重、曝險、淨值不變。"""
    cut = pd.Timestamp("2026-10-13")
    panel, sti = stage3["panel"], stage3["sti"]
    rng = np.random.default_rng(C.SEED)
    p2, s2 = panel.copy(), sti.copy()
    ap, as_ = p2.index > cut, s2.index > cut
    p2.loc[ap] = p2.loc[ap].values * rng.uniform(0.5, 1.5, p2.loc[ap].shape)
    s2.loc[as_] = s2.loc[as_].values * rng.uniform(0.5, 1.5, s2.loc[as_].shape)
    _, v2 = R3.build_all(p2, s2, stage3["frozen"])
    specs = R3.frozen_specs(stage3["frozen"]["params"])
    t2 = R3.run_groups(v2, {"S3": specs["S3"]}, v2.days[-1])
    v1, t1 = stage3["view"], stage3["test"]
    m = stage3["frozen"]["params"]["LASSO_MODEL"]
    k = int((v1.signal <= cut).sum())
    assert np.array_equal(v1.scores[m][:k], v2.scores[m][:k], equal_nan=True)
    a, b = t1["S3"], t2["S3"]
    assert np.array_equal(a["weights"].loc[:cut].values, b["weights"].loc[:cut].values)
    assert np.array_equal(a["targets"]["exposure"]["exposure"].values[:k], b["targets"]["exposure"]["exposure"].values[:k])
    assert np.array_equal(a["daily"]["nav"].loc[:cut].values, b["daily"]["nav"].loc[:cut].values)
    assert not np.allclose(a["daily"]["nav"].loc[cut:].iloc[1:].values, b["daily"]["nav"].loc[cut:].iloc[1:].values)


def test_assets_csv():
    path = os.path.join(C.TABLE_DIR, "assets_daily_close.csv")
    if not os.path.exists(path):
        pytest.skip("run_stage3.py 還沒跑")
    df = pd.read_csv(path)
    assert list(df.columns) == ["date", "ticker", "name", "UP", "P", "currency"]
    sig = pd.DatetimeIndex(pd.read_csv(os.path.join(C.TABLE_DIR, "test_scores.csv"), header=[0, 1], index_col=0).index)
    members = dl.membership_mask(sig, "in_index").any()
    assert set(df["ticker"]) == set(members.index[members.values]) and 30 <= len(set(df["ticker"])) <= 31   # 測試期成份股
    assert pd.to_datetime(df["date"]).max() <= pd.Timestamp(C.TEST_END)
    assert (df["currency"] == "SGD").all() and df["name"].notna().all()
    assert not df.duplicated(["date", "ticker"]).any()
