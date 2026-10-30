"""階段 1 相關檢查：前視擾動（因子與預處理）、日曆與 RZ_open、最大回撤，以及預處理的單元測試。"""
import numpy as np
import pandas as pd
import pytest

import config as C
import data_loader as dl
import factors as F
import metrics as M
import preprocess as PP


# ---------------------------------------------------------------- 10.1 前視偏誤擾動測試
def _pipeline(panel, sti, signal_lo, signal_hi):
    days = dl.trading_days(panel)
    tradable = dl.tradable_mask(panel, days)
    fac = F.compute_factors(panel, sti, days)
    sched = PP.weekly_schedule(days)
    sig = pd.DatetimeIndex(sched["signal"])
    sig = sig[(sig >= signal_lo) & (sig <= signal_hi)]
    z, _ = PP.preprocess_all(fac, tradable, dl.field(panel.loc[days], "MV"), sig)
    return days, tradable, fac, sched, z


@pytest.mark.parametrize("cut", ["2015-03-17", "2021-08-31", "2024-09-30"])   # 2021-08-31：之後 DBSL 下市、成份股變動（檢查 15）
def test_lookahead_perturbation(data, cut):
    panel, sti = data
    cut = pd.Timestamp(cut)
    rng = np.random.default_rng(C.SEED)
    after_p, after_s = panel.index > cut, sti.index > cut
    p2, s2 = panel.copy(), sti.copy()
    p2.loc[after_p] = p2.loc[after_p].values * rng.uniform(0.5, 1.5, size=p2.loc[after_p].shape)
    s2.loc[after_s] = s2.loc[after_s].values * rng.uniform(0.5, 1.5, size=s2.loc[after_s].shape)

    lo, hi = cut - pd.Timedelta(days=120), cut + pd.Timedelta(days=60)
    d1, t1, f1, sc1, z1 = _pipeline(panel, sti, lo, hi)
    d2, t2, f2, sc2, z2 = _pipeline(p2, s2, lo, hi)

    assert d1[d1 <= cut].equals(d2[d2 <= cut])
    assert t1.loc[:cut].equals(t2.loc[:cut])
    sc1, sc2 = sc1[sc1["signal"] <= cut], sc2[sc2["signal"] <= cut]
    assert sc1[["signal", "rebal"]].equals(sc2[["signal", "rebal"]])
    changed_after = 0
    for k in C.FACTORS:
        a, b = f1[k].loc[:cut].values, f2[k].loc[:cut].values
        assert np.array_equal(a, b, equal_nan=True), f"因子 {k} 在 {cut.date()} 之前被未來資料改變"
        za, zb = z1[k].loc[:cut].values, z2[k].loc[:cut].values
        assert len(za) > 0 and np.array_equal(za, zb, equal_nan=True), f"z_{k} 在 {cut.date()} 之前被未來資料改變"
        changed_after += not np.array_equal(f1[k].loc[cut:].iloc[1:].values, f2[k].loc[cut:].iloc[1:].values,
                                            equal_nan=True)
    assert changed_after >= len(C.FACTORS) - 1     # 擾動確實有作用（size 以外大多會變）


# ---------------------------------------------------------------- 10.2 日曆與 RZ_open
def test_schedule(data):
    panel, _ = data
    days = dl.trading_days(panel)
    sched = PP.weekly_schedule(days)
    assert (sched["rebal"] > sched["signal"]).all()
    pos = days.get_indexer(sched["rebal"])
    assert (days[pos - 1] == pd.DatetimeIndex(sched["signal"])).all()      # 訊號日 = 換倉日前一個交易日
    for reb in sched["rebal"]:
        tue = reb - pd.Timedelta(days=reb.day_of_week - C.REBALANCE_WEEKDAY)
        assert reb.day_of_week >= C.REBALANCE_WEEKDAY
        first = days[(days >= tue) & (days <= reb)]
        assert first[0] == reb                                             # 週二起第一個交易日
    # 有 週二–週五 交易日的週，都要有換倉
    cand = days[(days.day_of_week >= C.REBALANCE_WEEKDAY) & (days.day_of_week <= 4)]
    weeks = set(cand[days.get_indexer(cand) > 0].to_period("W-SUN"))
    assert weeks <= set(pd.DatetimeIndex(sched["rebal"]).to_period("W-SUN"))
    # 每週最多一次
    assert pd.DatetimeIndex(sched["rebal"]).to_period("W-SUN").is_unique


def test_rz_open_equals_rz_when_po_equals_p(data):
    panel, _ = data
    days = dl.trading_days(panel)
    rzo = PP.rz_open(panel, days)
    p = panel.loc[days]
    rz, po, px = dl.field(p, "RZ"), dl.field(p, "PO"), dl.field(p, "P")
    same = (po == px) & rz.notna()
    assert same.values.sum() > 1000
    assert np.allclose(rzo.values[same.values], rz.values[same.values], rtol=1e-12, atol=0)


def test_holding_returns_use_rebal_open():
    days = pd.bdate_range("2024-01-01", periods=15)
    rzo = pd.DataFrame({"A": np.arange(1.0, 16.0)}, index=days)
    sched = PP.weekly_schedule(days)
    hr = PP.holding_returns(rzo, sched)
    first = sched.iloc[0]
    assert first["rebal"].day_of_week == 1
    assert hr.iloc[0, 0] == pytest.approx(rzo.loc[first["next_rebal"], "A"] / rzo.loc[first["rebal"], "A"] - 1)
    assert np.isnan(hr.iloc[-1, 0])


def test_schedule_holiday_shift():
    days = pd.bdate_range("2024-01-01", "2024-01-31")
    days = days.drop([pd.Timestamp("2024-01-09"), pd.Timestamp("2024-01-23"), pd.Timestamp("2024-01-24")])
    sched = PP.weekly_schedule(days).set_index("rebal")
    assert pd.Timestamp("2024-01-10") in sched.index          # 週二假日 → 週三
    assert sched.loc["2024-01-10", "signal"] == pd.Timestamp("2024-01-08")
    assert pd.Timestamp("2024-01-25") in sched.index          # 週二、三都假日 → 週四


# ---------------------------------------------------------------- 10.5 最大回撤
def test_max_drawdown():
    assert M.max_drawdown(pd.Series([-0.10, 0.02])) == pytest.approx(-0.10)


# ---------------------------------------------------------------- 因子 / 預處理單元測試
def test_value_months_ago():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-05", "2024-04-04", "2024-04-05", "2024-04-08"])
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    out = F.value_months_ago(df, 3)["A"]
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.loc["2024-04-04"] == 1.0                         # 1/4 之前最近 = 1/2
    assert out.loc["2024-04-05"] == 2.0                         # 1/5 當天
    assert out.loc["2024-04-08"] == 2.0                         # 1/8 → 1/5


def test_rolling_mdd():
    rz = pd.DataFrame({"A": [100, 110, 99, 105, 120.0]})
    out = F.rolling_mdd(rz, n_ret=3)["A"]
    assert np.isnan(out.iloc[2])
    assert out.iloc[3] == pytest.approx(1 - 99 / 110)
    assert out.iloc[4] == pytest.approx(1 - 99 / 110)


def test_preprocess_neutralization():
    rng = np.random.default_rng(0)
    names = [f"S{i}" for i in range(30)]
    sector = pd.Series(np.repeat(["a", "b", "c"], 10), index=names)
    log_mv = pd.Series(rng.normal(size=30), index=names)
    raw = pd.Series(rng.normal(size=30), index=names) + log_mv
    z, st = PP.preprocess_cross_section(raw, sector, log_mv)
    assert st == "ok"
    assert z.mean() == pytest.approx(0, abs=1e-12) and z.std() == pytest.approx(1)
    for s in "abc":
        assert z[sector == s].mean() == pytest.approx(0, abs=1e-10)
    assert np.corrcoef(z, log_mv)[0, 1] == pytest.approx(0, abs=1e-10)
    # 少於 10 檔 → NaN
    z2, st2 = PP.preprocess_cross_section(raw.iloc[:9], sector, log_mv)
    assert st2 == "fewer_than_10" and z2.isna().all()
    # 樣本數 < 參數數 + 5 → NaN，不能退回未中性化
    z3, st3 = PP.preprocess_cross_section(raw.iloc[:12], pd.Series(list("abcdefghijkl"), index=names[:12]), log_mv)
    assert st3 == "neutralization_unavailable" and z3.isna().all()


# ---------------------------------------------------------------- ep_ttm、ivol60 等因子
def test_rolling_capm_matches_lstsq():
    rng = np.random.default_rng(1)
    n, w = 120, 60
    x = pd.Series(rng.normal(0, 0.01, n))
    y = pd.DataFrame({"A": 0.0005 + 1.2 * x + rng.normal(0, 0.005, n)})
    y.iloc[[70, 75, 90], 0] = np.nan                      # 缺值的日子不納入
    xm = pd.DataFrame({"A": x})
    alpha, sd, cnt = F.rolling_capm(y, xm, w, 48)
    for t in [59, 80, 119]:
        yy, xx = y["A"].iloc[t - w + 1:t + 1], x.iloc[t - w + 1:t + 1]
        ok = yy.notna()
        X = np.c_[np.ones(ok.sum()), xx[ok]]
        coef, *_ = np.linalg.lstsq(X, yy[ok].values, rcond=None)
        resid = yy[ok].values - X @ coef
        assert alpha.iloc[t, 0] == pytest.approx(coef[0], rel=1e-8, abs=1e-12)
        assert sd.iloc[t, 0] == pytest.approx(resid.std(ddof=1), rel=1e-8)
        assert cnt.iloc[t, 0] == ok.sum()
    assert np.isnan(alpha.iloc[58, 0])
    # 有效天數不足 80% → NaN
    y2 = y.copy(); y2.iloc[60:75, 0] = np.nan
    assert np.isnan(F.rolling_capm(y2, xm, w, 48)[0].iloc[80, 0])


def test_new_factors_on_real_data(data):
    panel, sti = data
    days = dl.trading_days(panel)
    fac = F.compute_factors(panel, sti, days)
    p = panel.loc[days]
    pe, eps, px, rz = (dl.field(p, k) for k in ["PE", "EPS", "P", "RZ"])
    pe5, eps5, p5 = pe.shift(5), eps.shift(5), px.shift(5)
    ep = fac["ep_ttm"].values
    pos = ((pe5 > 0) & fac["ep_ttm"].notna()).values
    assert pos.sum() > 50000
    assert np.allclose(ep[pos], (p5 / pe5 / px).values[pos])
    loss = (~(pe5 > 0) & eps5.notna() & fac["ep_ttm"].notna()).values
    listed_loss = (~(pe5 > 0) & eps5.notna() & rz.notna() & (dl.field(p, "MV") > 0)).values
    assert loss.sum() > 1000 and loss.sum() == listed_loss.sum() and (ep[loss] == 0).all()
    assert np.isnan(ep[(eps5.isna() & ~(pe5 > 0)).values]).all()
    # overnight20 + intraday20 = ln(RZ(t)/RZ(t−20))（兩者都有值時）
    both = (fac["overnight20"].notna() & fac["intraday20"].notna()).values
    total = (fac["overnight20"] + fac["intraday20"]).values[both]
    assert both.sum() > 50000 and np.allclose(total, np.log(rz / rz.shift(20)).values[both], atol=1e-10)
    # max20 ≥ 20 日內任何一天的報酬
    r = rz.pct_change(fill_method=None)
    m = fac["max20"].notna().values
    assert (fac["max20"].values[m] >= r.values[m] - 1e-15).all()


def test_dy_nan_first_year_after_listing(data):
    panel, sti = data
    days = dl.trading_days(panel)
    dy = F.dividends_ttm(panel, days)
    rz = dl.field(panel.loc[days], "RZ")
    for t in ["YAFH", "CAP1", "KEDC", "FRAL", "MTRE"]:
        listed = rz[t].first_valid_index()
        assert dy[t].loc[:listed + pd.DateOffset(years=1) - pd.Timedelta(days=1)].isna().all()
        assert dy[t].loc[listed + pd.DateOffset(years=1):].notna().any()
    # 資料開始前就上市的股票，維持「資料第一年 NaN」的規則
    assert dy["DBSS"].loc["2011-01-04":].notna().all() and dy["DBSS"].loc[:"2010-12-31"].isna().all()
