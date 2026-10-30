"""股票池在回測中的檢查：剔除與納入、下市轉現金。前視擾動涵蓋下市與成份股變動的部分，
在 test_stage2.py 的 test_lookahead_perturbation_stage2 加了 2021-08-31 這個切點。"""
import numpy as np
import pandas as pd
import pytest

import backtest as BT
import config as C
import data_loader as dl

from test_stage2 import ctx, main_results   # noqa: F401（共用 session fixture）

GROUPS = ["B0'", "B1", "A1", "A2", "S1", "S2", "S3"]


@pytest.fixture(scope="module")
def member(ctx):
    return dl.membership_mask(ctx.signal, "in_index").values


def _signal_of(ctx, rebal):
    return ctx.signal[ctx.rebal.get_indexer(pd.DatetimeIndex(rebal))]


# ---------------------------------------------------------------- 14. 下市（小例子手算）
def test_delisting_converts_to_cash_hand_calc():
    days = pd.bdate_range("2024-01-01", periods=7)
    rz = np.array([[100, 100], [101, 102], [102, 104], [103, 103], [104, np.nan], [105, np.nan], [106, np.nan]])
    px = rz.copy(); po = rz.copy(); vo = np.where(np.isfinite(rz), 1.0, 0.0)
    mkt = BT.Market(days, rz, po, px, vo, ["A", "B"], delisted={"B": days[3]})   # B 在第 3 天收盤下市
    res = BT.run(mkt, [days[1], days[5]], np.array([[0.5, 0.5], [0.5, 0.5]]), cost_bps=0, band=None,
                 stock_cap=None, sector=np.zeros(2, int))
    d = res["daily"]
    nav3 = 0.5 * rz[3, 0] / rz[1, 0] + 0.5 * rz[3, 1] / rz[1, 1]
    assert np.isclose(d["nav"].loc[days[3]], nav3)
    wa, wb = 0.5 * rz[3, 0] / rz[1, 0] / nav3, 0.5 * rz[3, 1] / rz[1, 1] / nav3
    assert np.isclose(d["cash"].loc[days[3]], wb)                                  # L_i 收盤轉現金，淨值不跳動
    assert np.isclose(d["ret"].loc[days[4]], wa * (rz[4, 0] / rz[3, 0] - 1))      # 之後現金報酬 = 0
    w = res["weights"]
    assert w.loc[days[5], "B"] == 0 and np.isclose(w.loc[days[5], "A"], 1.0)      # 現金在下次換倉重新配置
    tr = res["trades"]
    b = tr[tr.ticker == "B"].iloc[0]
    assert b["delisted"] and b["sell_date"] == days[3] and np.isclose(b["pnl"], rz[3, 1] / rz[1, 1] - 1)


# ---------------------------------------------------------------- 13. 剔除與納入
def test_no_weight_outside_universe(ctx, main_results, member):
    for g in GROUPS:
        W = main_results[g]["weights"]
        j = ctx.rebal.get_indexer(W.index)
        out = (W.values > 0) & ~member[j]
        ok = ~ctx.mkt.exec_ok[ctx.days.get_indexer(W.index)]             # 賣不掉的例外
        bad = out & ~ok
        assert not bad.any(), (g, W.index[bad.any(axis=1)][:5].tolist())


def test_excluded_sold_first_rebal(ctx, main_results):
    per = dl.membership_periods()
    exits = per[per["end"].notna() & (per["end"] >= ctx.signal[ctx.j0])]
    rows = []
    for g in GROUPS:
        W = main_results[g]["weights"]
        for _, r in exits.iterrows():
            t = r["ds_ticker"]
            before = W.index[_signal_of(ctx, W.index) <= r["end"]]
            after = W.index[_signal_of(ctx, W.index) > r["end"]]
            if not len(before) or not len(after) or W.loc[before[-1], t] <= 0:
                continue
            first = after[0]
            held = W.loc[first, t] > 0
            stuck = not ctx.mkt.exec_ok[ctx.days.get_loc(first), ctx.tickers.index(t)]
            rows.append((g, t, first.date(), held, stuck))
            assert not held or stuck, (g, t, first)
    print("\n剔除後第一次換倉（組別、股票、換倉日、仍持有、賣不掉）：", rows[:20])


# ---------------------------------------------------------------- 14. 下市（實際資料）
def test_dbsl_delisting(ctx, main_results):
    L = pd.Timestamp("2021-09-09")
    i = ctx.tickers.index("DBSL")
    for g in GROUPS:
        res = main_results[g]
        W = res["weights"]
        assert (W.loc[W.index > L, "DBSL"] == 0).all(), g
        d = res["daily"]
        prev_reb = W.index[W.index <= L][-1]
        if W.loc[prev_reb, "DBSL"] <= 0:
            continue
        # L_i 收盤的漂移權重 = 轉入現金的金額：w_L = w_new × NAV_開盤成交後 × RZ(L)/RZ_open(換倉日) / NAV(L)
        t0, tL = ctx.days.get_loc(prev_reb), ctx.days.get_loc(L)
        wn = W.loc[prev_reb].values
        nav_after = d["nav"].loc[prev_reb] / (1 + wn @ ctx.mkt.r_id[t0])
        w_L = wn[i] * nav_after * ctx.mkt.rz[tL, i] / ctx.mkt.rzo[t0, i] / d["nav"].loc[L]
        jump = d["cash"].loc[L] - d["cash"].loc[ctx.days[tL - 1]] / (1 + d["ret"].loc[L])
        assert w_L > 0 and np.isclose(jump, w_L, rtol=1e-8), (g, jump, w_L)
        nxt = ctx.days[tL + 1]
        assert np.isclose(d["cash"].loc[nxt], d["cash"].loc[L] / (1 + d["ret"].loc[nxt])), g   # 之後報酬 = 0
        tr = res["trades"]
        row = tr[(tr.ticker == "DBSL") & tr["delisted"].fillna(False).astype(bool)]
        assert len(row) == 1 and row["sell_date"].iloc[0] == L, g


def test_sgps_not_held_to_delisting(ctx, main_results):
    out = pd.Timestamp("2020-06-21")
    for g in GROUPS:
        W = main_results[g]["weights"]
        sig = _signal_of(ctx, W.index)
        late = W.index[sig > out]
        held = W.loc[late, "SGPS"] > 0
        assert not held.any(), (g, late[held.values][:3].tolist())
        tr = main_results[g]["trades"]
        if len(tr):
            assert not (tr[tr.ticker == "SGPS"]["delisted"].fillna(False).astype(bool)).any(), g
