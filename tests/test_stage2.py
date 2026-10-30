"""階段 2 的檢查：前視擾動（延伸到分數、權重、曝險、淨值）、換倉時點、權重限制、回測手算、Lasso 訓練資料時間、λ 驗證段、回看資料截止日、產業上限基準、資料截斷。"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import backtest as BT
import config as C
import data_loader as dl
import model as MD
import portfolio as P
import run_stage2 as R2
import volatility as V


@pytest.fixture(scope="session")
def ctx():
    panel, sti = R2.load_truncated()
    c = R2.Context(panel, sti, verbose=False)
    c.fit_models(which=("lasso260",))
    return c


@pytest.fixture(scope="session")
def main_results(ctx):
    groups = {
        "B0'": R2.spec(universe="all", weighting="mv", stock_cap=None, sector_over=None, band=None),
        "B1": R2.spec(universe="all", weighting="ew"),
        "A1": R2.spec(universe="all", weighting="invvol"),
        "A2": R2.with_exp(R2.spec(universe="all", weighting="invvol")),
        "S1": R2.spec(weighting="ew"),
        "S2": R2.spec(weighting="invvol"),
        "S3": R2.with_exp(R2.spec(weighting="invvol")),
    }
    return {g: R2.run_strategy(ctx, s) for g, s in groups.items()}


# ---------------------------------------------------------------- 10.11 資料截斷
def test_truncation(ctx):
    assert ctx.days[-1] <= pd.Timestamp(C.FREEZE_DATE)
    assert ctx.panel.index[-1] <= pd.Timestamp(C.FREEZE_DATE)
    assert ctx.sti.index[-1] <= pd.Timestamp(C.FREEZE_DATE)
    assert ctx.rebal[-1] <= pd.Timestamp(C.FREEZE_DATE)


def _max_date(df):
    best = None
    cands = [df.index] if isinstance(df.index, pd.DatetimeIndex) else []
    if isinstance(df.index, pd.MultiIndex):
        cands += [df.index.get_level_values(i) for i in range(df.index.nlevels)]
    cands += [df[c] for c in df.columns]
    for s in cands:
        s = pd.Series(s)
        if s.dtype == object or "datetime" in str(s.dtype):
            d = pd.to_datetime(s, errors="coerce", format="mixed") if s.dtype == object else s
            if d.notna().mean() > 0.5:
                m = d.max()
                best = m if best is None or m > best else best
    return best


def test_stage2_outputs_end_before_freeze():
    path = os.path.join(C.LOG_DIR, "stage2_manifest.json")
    if not os.path.exists(path):
        pytest.skip("run_stage2.py 還沒跑")
    files = json.load(open(path))
    checked = 0
    for f in files:
        full = os.path.join(C.ROOT, f)
        if f.endswith(".csv"):
            df = pd.read_csv(full, index_col=0)
            try:
                df.index = pd.to_datetime(df.index)
            except (ValueError, TypeError):
                df = df.reset_index()
        elif f.endswith(".parquet"):
            df = pd.read_parquet(full)
        else:
            continue
        m = _max_date(df)
        if m is not None:
            checked += 1
            assert m <= pd.Timestamp(C.FREEZE_DATE), f"{f} 有 {m.date()} 的資料"
    assert checked >= 10


# ---------------------------------------------------------------- 10.6、10.7 Lasso 訓練資料與 λ 驗證
def test_lasso_training_ends_before_signal(ctx, capsys):
    log = ctx.model_logs["lasso260"]
    sig = pd.to_datetime(log["signal"]); last = pd.to_datetime(log["last_y_end"])
    assert (last <= sig).all()
    # 獨立重算：最後一筆 y 的期數 = 訊號日前兩期（上週換倉 → 本週換倉那一期不能用）
    sched = ctx.sched
    for _, row in log.iterrows():
        j = int(np.where(ctx.signal == row["signal"])[0][0])
        el = MD.eligible_periods(ctx.signal.values, ctx.y_end, j)
        assert el[-1] == j - 2
        assert sched.loc[j - 1, "next_rebal"] > row["signal"]          # 上一期的報酬要到本週換倉日開盤才確定
    with capsys.disabled():
        print("\n  訊號日 / 訓練資料最後一筆 y 的結束日（換倉日開盤）")
        for i in np.linspace(0, len(log) - 1, 6).astype(int):
            print(f"  {log['signal'].iloc[i].date()}  {pd.Timestamp(log['last_y_end'].iloc[i]).date()}")


def test_lambda_validation_inside_window(ctx):
    log = ctx.model_logs["lasso260"]
    y_end = pd.DatetimeIndex(ctx.y_end)
    for _, row in log[log["n_screened"] > 0].iterrows():                # 沒有因子通過篩選的期數不選 λ
        assert row["val_first_idx"] >= row["first_y_start_idx"]
        assert row["val_last_idx"] - row["val_first_idx"] + 1 == C.LAMBDA_VALID_WEEKS
        assert y_end[row["val_last_idx"]] <= row["signal"]
        assert row["val_last_y_end"] == row["last_y_end"]               # 驗證段就是窗口最後 104 期


def test_ic_screening(ctx):
    """IC 篩選：只有通過的因子可以有係數。
    static：清單 = 訓練期（訊號日 ≤ TRAIN_END）週 IC 的 NW t > 門檻，跟階段 1 的 IC 表一致。
    rolling：t 值只用當期訓練窗口（持有期結束 ≤ 訊號日）的週 IC。"""
    log, coef = ctx.model_logs["lasso260"], ctx.model_coefs["lasso260"]
    if C.SCREEN_MODE == "static":
        tab = pd.read_csv(os.path.join(C.TABLE_DIR, "ic_summary_train.csv"), index_col=0)
        assert ctx.screen == [k for k in ctx.factor_names if tab.loc[k, "NW_t"] > C.SCREEN_T_MIN]
        assert np.allclose(ctx.screen_table["NW_t"].astype(float).values,
                           tab.loc[ctx.factor_names, "NW_t"].values, rtol=0, atol=1e-10)
        out = [k for k in ctx.factor_names if k not in ctx.screen]
        assert (coef[[ctx.factor_names.index(k) for k in out]].values == 0).all()
        return
    ic = MD.weekly_factor_ic(ctx.X, ctx.y, ctx.tradable)
    F = len(ctx.factor_names)
    for i in np.linspace(0, len(log) - 1, 5).astype(int):
        row = log.iloc[i]
        j = int(np.where(ctx.signal == row["signal"])[0][0])
        win = MD.eligible_periods(ctx.signal.values, ctx.y_end, j)[-C.LASSO_WINDOW:]
        assert win[0] == row["first_y_start_idx"] and ctx.y_end[win[-1]] <= row["signal"]
        keep, t = MD.screen_factors(ic, win)
        assert np.allclose(t, [row[f"t_{k}"] for k in range(F)], equal_nan=True)
        assert list(keep) == list(row["screen_idx"]) and (t[keep] > C.SCREEN_T_MIN).all()
        beta = coef.loc[row["signal"]].values
        assert (beta[np.setdiff1d(np.arange(F), keep)] == 0).all()


# ---------------------------------------------------------------- 10.3 權重
def test_weights_sum_nonneg_cap(main_results):
    for g, res in main_results.items():
        W, rb = res["weights"], res["rebal"]
        assert (W.values >= -1e-12).all(), g
        assert np.allclose(W.sum(axis=1).values, rb["exposure_target"].values, atol=1e-8), g
        if g != "B0'":
            assert W.values.max() <= C.MAX_WEIGHT + 1e-8, g
        assert (rb["frozen_over_cap"] == 0).all()


def test_sector_caps_hold(ctx, main_results):
    tg = main_results["S2"]["targets"]
    for k in range(len(tg["rel"])):
        w = tg["rel"][k]
        caps = tg["caps"][k]
        relaxed = tg["info"][k]["cap_status"] != "ok"
        for s in range(len(P.SECTORS)):
            if not relaxed:
                assert w[ctx.sec == s].sum() <= caps[s] + 1e-8


# ---------------------------------------------------------------- 10.4 手算回測
def test_backtest_hand_calc():
    days = pd.bdate_range("2024-01-01", periods=10)
    rng = np.random.default_rng(3)
    rz = 100 * np.cumprod(1 + rng.normal(0, 0.01, (10, 3)), axis=0)
    px = rz / 2
    po = px * (1 + rng.normal(0, 0.003, (10, 3)))
    vo = np.ones((10, 3))
    mkt = BT.Market(days, rz, po, px, vo, ["A", "B", "C"])
    res = BT.run(mkt, [days[1]], np.array([[1 / 3, 1 / 3, 1 / 3]]), cost_bps=0, band=None, stock_cap=None,
                 sector=np.zeros(3, int))
    rzo = rz * po / px
    nav_hand = (rz[1:] / rzo[1]).mean(axis=1)               # 第 1 天開盤以 RZ_open 買進，之後買進持有
    assert np.allclose(res["daily"]["nav"].values, nav_hand, rtol=1e-12)
    # 成交價用換倉日的 PO：開盤前那一段報酬不計入（從現金開始）
    assert np.isclose(res["daily"]["ret"].iloc[0], (rz[1] / rzo[1]).mean() - 1)


def test_backtest_cost_and_split():
    days = pd.bdate_range("2024-01-01", periods=6)
    rz = np.array([[100, 100], [101, 99], [102, 98], [103, 99], [104, 100], [105, 101]], float)
    px = rz.copy(); po = rz * 0.995; vo = np.ones_like(rz)
    mkt = BT.Market(days, rz, po, px, vo, ["A", "B"])
    rel = np.array([[1.0, 0.0], [0.0, 1.0]])
    res = BT.run(mkt, [days[1], days[3]], rel, cost_bps=15, band=None, stock_cap=None, sector=np.zeros(2, int))
    d = res["daily"]
    rzo = rz * po / px
    nav1 = (1 - 2 * 15e-4) * rz[1, 0] / rzo[1, 0]                         # 建倉：現金 1 → 股票 1，Σ|Δw| = 2
    assert np.isclose(d["nav"].iloc[0], nav1)
    nav2 = nav1 * rz[2, 0] / rz[1, 0]
    nav3_open = nav2 * rzo[3, 0] / rz[2, 0]
    nav3 = nav3_open * (1 - 2 * 15e-4) * rz[3, 1] / rzo[3, 1]
    assert np.isclose(d["nav"].iloc[2], nav3)


def test_suspended_stock_frozen():
    days = pd.bdate_range("2024-01-01", periods=5)
    rz = np.array([[100, 100], [101, 100], [102, 100], [103, 101], [104, 102]], float)
    px = rz.copy(); po = rz.copy(); vo = np.ones_like(rz)
    po[2, 1] = np.nan; vo[2, 1] = 0                                            # 第 2 天 B 停牌
    mkt = BT.Market(days, rz, po, px, vo, ["A", "B"])
    res = BT.run(mkt, [days[1], days[2]], np.array([[0.5, 0.5], [1.0, 0.0]]), cost_bps=0, band=None,
                 stock_cap=None, sector=np.zeros(2, int))
    W = res["weights"]
    assert W.iloc[1, 1] > 0.4 and res["rebal"]["n_frozen"].iloc[1] == 1      # B 不能賣，維持原權重
    assert np.isclose(W.iloc[1].sum(), 1.0)


def test_suspended_sell_skips_lowest_buy():
    """選股組：要賣的持股停牌賣不掉 → 少買分數最低的新股，持股數維持 K。"""
    days = pd.bdate_range("2024-01-01", periods=5)
    rz = 100 + np.cumsum(np.ones((5, 4)), axis=0)
    px = rz.copy(); po = rz.copy(); vo = np.ones_like(rz)
    po[2, 0] = np.nan; vo[2, 0] = 0                                            # 第 2 天 A 停牌
    mkt = BT.Market(days, rz, po, px, vo, ["A", "B", "C", "D"])
    rel = np.array([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]])                       # K = 2：第二次要賣 A、B，買 C、D
    pr = np.array([[2, 1, 0, 0], [0, 0, 2.0, 1.0]])                            # D 的分數比 C 低
    res = BT.run(mkt, [days[1], days[2]], rel, cost_bps=0, band=None, stock_cap=None,
                 sector=np.zeros(4, int), priority=pr)
    w = res["weights"].iloc[1]
    assert w["A"] > 0 and w["B"] == 0 and w["C"] > 0 and w["D"] == 0           # 保留 A、賣 B、只買 C
    assert (w > 0).sum() == 2 and np.isclose(w.sum(), 1.0)
    assert res["rebal"]["n_skipped_buy"].iloc[1] == 1
    # 不給 priority（非選股組）：照原規則，A 保留、C 和 D 都買
    res0 = BT.run(mkt, [days[1], days[2]], rel, cost_bps=0, band=None, stock_cap=None, sector=np.zeros(4, int))
    assert (res0["weights"].iloc[1] > 0).sum() == 3


# ---------------------------------------------------------------- 10.8 回看資料截止在訊號日
def _perturb_after(a, pos, seed=0):
    b = a.copy()
    rng = np.random.default_rng(seed)
    b[pos + 1:] = b[pos + 1:] * rng.uniform(0.5, 1.5, b[pos + 1:].shape) + rng.normal(0, 0.01, b[pos + 1:].shape)
    return b


def test_sigma_lookback(ctx):
    pos = int(ctx.sig_pos[ctx.j0 + 300])
    r2 = _perturb_after(ctx.r, pos)
    ph2, pl2 = _perturb_after(ctx.ph, pos), _perturb_after(ctx.pl, pos)
    for m in V.STOCK_METHODS:
        a = V.stock_sigma(ctx.r, [pos], m, ph=ctx.ph, pl=ctx.pl, traded=ctx.traded)
        b = V.stock_sigma(r2, [pos], m, ph=ph2, pl=pl2, traded=ctx.traded)
        assert np.array_equal(a, b, equal_nan=True), m
    w = np.zeros(ctx.N); w[np.where(ctx.tradable[ctx.j0 + 300])[0][:12]] = 1 / 12
    for m in V.PORT_METHODS:
        ya = V.portfolio_returns(ctx.r, pos, w, 500); yb = V.portfolio_returns(r2, pos, w, 500)
        assert V.forecast_port_sigma(ya, m) == V.forecast_port_sigma(yb, m), m


# ---------------------------------------------------------------- 10.10 產業上限的基準權重用訊號日的 MV
def test_sector_bench_uses_signal_mv(ctx):
    mv = dl.field(ctx.panel.loc[ctx.days], "MV")
    assert np.allclose(ctx.mv, mv.reindex(ctx.signal).values, equal_nan=True)
    assert not np.allclose(ctx.mv, mv.reindex(ctx.rebal).values, equal_nan=True)
    j = ctx.j0 + 100
    b1 = P.bench_sector_weights(ctx.mv[j], ctx.tradable[j], ctx.sec)
    m = np.where(ctx.tradable[j] & np.isfinite(ctx.mv[j]), ctx.mv[j], 0.0)
    assert np.isclose(b1[ctx.sec[0]], m[ctx.sec == ctx.sec[0]].sum() / m.sum())


# ---------------------------------------------------------------- 10.1 前視偏誤擾動（延伸到分數、權重、曝險、淨值）
def _run_pipeline(panel, sti, screen=None):
    c = R2.Context(panel, sti, verbose=False, screen=screen)
    c.fit_models(which=("lasso260",))
    out = {"screen": c.screen, "z": c.z, "scores": pd.DataFrame(c.scores["lasso260"], index=c.signal),
           "sigma": pd.DataFrame(c.sigma("ewma94"), index=c.signal)}
    for g, s in {"S3": R2.with_exp(R2.spec()), "A2": R2.with_exp(R2.spec(universe="all"))}.items():
        res = R2.run_strategy(c, s)
        n = len(res["targets"]["rel"])
        sig = c.signal[c.j0:c.j0 + n]
        out[g] = {"rel": pd.DataFrame(res["targets"]["rel"], index=sig),
                  "exposure": pd.Series(res["targets"]["exposure"]["exposure"].values, index=sig),
                  "sigma_p": pd.Series(res["targets"]["exposure"]["sigma_p"].values, index=sig),
                  "weights": res["weights"], "nav": res["daily"]["nav"]}
    return out


@pytest.mark.parametrize("cut", ["2019-06-14", "2021-08-31", "2024-06-14"])   # 2021-08-31：之後 DBSL 下市、成份股變動（檢查 15）
def test_lookahead_perturbation_stage2(cut):
    cut = pd.Timestamp(cut)
    panel, sti = R2.load_truncated()
    end = cut + pd.Timedelta(days=120)
    panel, sti = panel.loc[:end], sti.loc[:end]
    rng = np.random.default_rng(C.SEED)
    p2, s2 = panel.copy(), sti.copy()
    ap, as_ = p2.index > cut, s2.index > cut
    p2.loc[ap] = p2.loc[ap].values * rng.uniform(0.5, 1.5, p2.loc[ap].shape)
    s2.loc[as_] = s2.loc[as_].values * rng.uniform(0.5, 1.5, s2.loc[as_].shape)
    # 固定篩選（static）的因子清單是事先凍結的輸入，兩次都用同一份；
    # 它用到 2011–2022 整段 IC 造成的前視偏誤是已知、已揭露的設計選擇，這個測試檢查的是其他環節。
    a = _run_pipeline(panel, sti, screen=R2.Context(*R2.load_truncated(), verbose=False).screen)
    b = _run_pipeline(p2, s2, screen=a["screen"])

    def same(x, y, what):
        x, y = x.loc[:cut], y.loc[:cut]
        assert len(x) > 0 and x.index.equals(y.index), what
        assert np.array_equal(x.values, y.values, equal_nan=True), f"{what} 在 {cut.date()} 以前被未來資料改變"

    for k in C.FACTORS:
        same(a["z"][k], b["z"][k], f"z_{k}")
    same(a["scores"], b["scores"], "Lasso 分數")
    same(a["sigma"], b["sigma"], "σ̂")
    for g in ["S3", "A2"]:
        for key in ["rel", "exposure", "sigma_p", "weights", "nav"]:
            same(a[g][key], b[g][key], f"{g} {key}")
        # 擾動確實改變了 cut 之後的淨值
        assert not np.allclose(a[g]["nav"].loc[cut:].iloc[1:].values, b[g]["nav"].loc[cut:].iloc[1:].values)
