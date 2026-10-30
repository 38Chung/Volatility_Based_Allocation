"""月頻對照實驗的必要檢查。"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

import config as C
import portfolio as P
import run_freq_compare as FC
import run_stage3 as R3
import volatility as V


@pytest.fixture(scope="session")
def frozen():
    with open(os.path.join(C.OUT_DIR, "frozen_params.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def built(frozen):
    return FC.build(frozen["params"])


@pytest.fixture(scope="session")
def fc_table():
    return pd.read_csv(os.path.join(C.TABLE_DIR, "freq_compare.csv"))


# ---------------------------------------------------------------- 月頻換倉日
def test_monthly_calendar(built):
    ctx, mv = built
    wreb = pd.DatetimeIndex(ctx.sched["rebal"])
    first_of_month = pd.Series(wreb, index=wreb).groupby(wreb.to_period("M")).min()
    assert list(mv.rebal) == list(first_of_month.values)
    assert mv.rebal.to_period("M").is_unique                                   # 每個曆月最多一次
    pos = ctx.days.get_indexer(mv.rebal)
    assert (pos > 0).all()
    assert (mv.signal == ctx.days[pos - 1]).all()                              # 訊號日 = 換倉日前一個交易日
    assert (mv.signal < mv.rebal).all()


# ---------------------------------------------------------------- 月頻 Lasso 的訓練資料時間
def test_monthly_lasso_timing(built, frozen):
    ctx, mv = built
    log = mv.model_logs[frozen["params"]["LASSO_MODEL"]]
    assert (log["n_periods"] <= FC.MONTHLY["window"]).all()
    assert (log["n_periods"] >= FC.MONTHLY["min_window"]).all()
    for _, row in log.iterrows():
        t = pd.Timestamp(row["signal"])
        assert pd.Timestamp(row["last_y_end"]) <= t
        assert pd.Timestamp(row["val_last_y_end"]) <= t
        assert row["val_first_idx"] >= row["first_y_start_idx"]                # 驗證段在訓練窗口之內
        assert row["val_last_idx"] - row["val_first_idx"] + 1 == FC.MONTHLY["val_periods"]
    pick = log.iloc[np.linspace(0, len(log) - 1, 6).astype(int)]
    print("\n訊號日      訓練資料最後一筆 y 的結束日   選 λ 驗證段最後結束日")
    for _, row in pick.iterrows():
        print(f"{pd.Timestamp(row['signal']).date()}  {pd.Timestamp(row['last_y_end']).date()}"
              f"                  {pd.Timestamp(row['val_last_y_end']).date()}")


# ---------------------------------------------------------------- σ̂ᵢ、GARCH 回看資料截止在訊號日
def test_lookback_ends_at_signal(built, frozen):
    ctx, mv = built
    params = frozen["params"]
    k = mv.j0 + 60
    pos = mv.sig_pos[k]
    rng = np.random.default_rng(0)
    r2 = mv.r.copy()
    r2[pos + 1:] *= rng.uniform(0.5, 1.5, r2[pos + 1:].shape)
    ph2, pl2 = mv.ph.copy(), mv.pl.copy()
    ph2[pos + 1:] *= 1.3; pl2[pos + 1:] *= 0.7
    m = params["SIGMA_METHOD"]
    a = V.stock_sigma(mv.r, mv.sig_pos[k:k + 1], m, ph=mv.ph, pl=mv.pl, traded=mv.traded)
    b = V.stock_sigma(r2, mv.sig_pos[k:k + 1], m, ph=ph2, pl=pl2, traded=mv.traded)
    assert np.array_equal(a, b, equal_nan=True)
    spec = FC.monthly_specs(params)["S3"]["exposure"]
    w = np.where(mv.tradable[k], 1.0, 0.0); w /= w.sum()
    ea = P.exposures(spec, w[None, :], mv.r, mv.sig_pos[k:k + 1], mv.sti_ret)
    eb = P.exposures(spec, w[None, :], r2, mv.sig_pos[k:k + 1], mv.sti_ret)
    assert np.isfinite(ea["sigma_p"].iloc[0])
    assert ea["sigma_p"].iloc[0] == eb["sigma_p"].iloc[0]


# ---------------------------------------------------------------- 資料截斷
def test_outputs_end_by_freeze(built, fc_table):
    ctx, mv = built
    assert ctx.days[-1] <= pd.Timestamp(C.FREEZE_DATE)
    assert mv.rebal[-1] <= pd.Timestamp(C.FREEZE_DATE)
    assert pd.to_datetime(fc_table["end"]).max() <= pd.Timestamp(C.FREEZE_DATE)
    assert set(fc_table["period"]) == {"train", "valid", "full"}               # 不跑測試期


# ---------------------------------------------------------------- 參數跟凍結檔一致
def test_params_match_frozen(built, frozen):
    ctx, mv = built
    params = frozen["params"]
    wk = R3.frozen_specs(params)
    mo = FC.monthly_specs(params)
    for g, s in mo.items():
        s = dict(s)
        if s["exposure"] is not None:
            e = dict(s["exposure"]); assert e.pop("horizon") == 21
            s["exposure"] = e
        assert s == wk[g], g                                                    # 除了 GARCH 預測天數，全部相同
    assert mv.screen == params["SCREEN_FACTORS"]
    assert FC.MONTHLY == dict(window=60, min_window=36, val_periods=24, garch_horizon=21)
    # 凍結檔沒有被改：跟凍結當下的複本相同
    snap = os.path.join(C.LOG_DIR, "frozen_params_snapshot.json")   # 階段 2 凍結當下的複本
    h = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    assert h(os.path.join(C.OUT_DIR, "frozen_params.json")) == h(snap)


# ---------------------------------------------------------------- 週頻那一欄 = perf_frozen_history
def test_weekly_matches_frozen_history(fc_table):
    p = pd.read_csv(os.path.join(C.TABLE_DIR, "perf_frozen_history.csv"))
    w = fc_table[fc_table.freq == "weekly"].merge(p, on=["portfolio", "period", "cost"], suffixes=("", "_p"))
    assert len(w) == 6 * 3 * 2
    for c in ["ann_return", "ann_vol", "sharpe", "max_drawdown"]:
        assert np.allclose(w[c], w[c + "_p"], rtol=0, atol=1e-12), c
    assert np.allclose(w["avg_turnover_per_rebal"], w["avg_weekly_turnover"], rtol=0, atol=1e-12)
