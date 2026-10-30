"""股票池的檢查：成份股遮罩、下市（資料部分）、讀檔規則。"""
import numpy as np
import pandas as pd
import pytest

import config as C
import data_loader as dl
import preprocess as PP


@pytest.fixture(scope="module")
def cal(data):
    panel, _ = data
    days = dl.trading_days(panel)
    sched = PP.weekly_schedule(days)
    sched = sched[sched["signal"] >= C.START_SIGNAL].reset_index(drop=True)
    return days, sched, pd.DatetimeIndex(sched["signal"])


# ---------------------------------------------------------------- 2.1 讀檔
def test_panel_universe_and_last_day(data):
    panel, sti = data
    assert sorted(panel.columns.get_level_values("ticker").unique()) == sorted(C.TICKERS) and len(C.TICKERS) == 50
    vo = dl.field(panel, "VO")
    assert panel.index[-1] == vo.index[(vo > 0).any(axis=1)][-1] == pd.Timestamp("2026-10-29")   # 10/30 空列已刪
    assert sti.index.equals(panel.index) and list(sti.columns) == ["PI", "RI", "DY", "MV"]
    assert sti["RI"].notna().mean() > 0.99


# ---------------------------------------------------------------- 12. 成份股遮罩
def test_membership_count(cal):
    _, _, sig = cal
    n = dl.membership_mask(sig, "in_index").sum(axis=1)
    yafh = (sig >= "2022-04-28") & (sig <= "2022-05-04")      # 揚子江金融分拆後暫時留在指數 → 31 檔
    assert (n[~yafh] == 30).all() and (n[yafh] == 31).all()


def test_membership_events_switch_on_effective_date(cal):
    _, _, sig = cal
    per = dl.membership_periods()
    m = dl.membership_mask(sig, "in_index")
    for _, r in per.iterrows():
        t = r["ds_ticker"]
        if r["start"] > sig[0]:
            first_new = sig[sig >= r["start"]][0]
            last_old = sig[sig < r["start"]][-1]
            assert m.at[first_new, t] and not m.at[last_old, t], (t, r["start"])
        if pd.notna(r["end"]) and r["end"] >= sig[0]:
            last_in = sig[sig <= r["end"]][-1]
            first_out = sig[sig > r["end"]][0]
            assert m.at[last_in, t] and not m.at[first_out, t], (t, r["end"])


def test_tradable_only_members(data, cal):
    panel, _ = data
    days, _, sig = cal
    tr = dl.tradable_mask(panel, days).reindex(sig)
    m = dl.membership_mask(sig, "in_index")
    assert not (tr & ~m).values.any()
    assert (tr.sum(axis=1) >= 26).all()


# ---------------------------------------------------------------- 14. 下市（資料部分）
def test_delisted_nan_after_last_trade(data):
    panel, _ = data
    for t, d in C.DELISTED.items():
        L = pd.Timestamp(d)
        after = panel.loc[panel.index > L, t]
        assert after.isna().all().all(), t
        vo = panel[(t, "VO")]
        assert vo[vo > 0].index[-1] == L, t                                     # L_i = 最後一個 VO > 0
        assert panel.loc[L, (t, "RZ")] == panel.loc[L, (t, "RZ")]               # L_i 當天仍有值


def test_tradable_false_after_delisting(data, cal):
    panel, _ = data
    days, _, sig = cal
    tr = dl.tradable_mask(panel, days)
    for t, d in C.DELISTED.items():
        assert not tr.loc[tr.index > pd.Timestamp(d), t].any(), t
    # DBSL：2021-09-06（週一）開盤由 CAP1 取代，9/9 才停止交易；
    # 9/6 訊號日已經不在股票池，9/7 換倉日開盤賣出，不會持有到 L_i
    m = dl.membership_mask(sig, "in_index")
    assert m.at[pd.Timestamp("2021-08-30"), "DBSL"] and not m.loc[sig >= "2021-09-06", "DBSL"].any()
    assert m.at[pd.Timestamp("2021-09-06"), "CAP1"] and not tr.at[pd.Timestamp("2021-09-06"), "CAP1"]   # CAP1 9/20 才開始交易
    # SGPS 2020-06-21 離開指數，之後到下市（2022-04-07）都不在股票池
    late = sig[(sig > "2020-06-21") & (sig <= "2022-04-07")]
    assert not m.loc[late, "SGPS"].any() and not tr.reindex(late)["SGPS"].any()


def test_holding_return_to_delisting_close(data, cal):
    panel, _ = data
    days, sched, _ = cal
    rzo = PP.rz_open(panel, days)
    rz = dl.field(panel.loc[days], "RZ")
    hr = PP.holding_returns(rzo, sched, 1, rz=rz)
    s = pd.Timestamp("2021-09-06")                     # 換倉日 9/7 開盤 → 下次換倉 9/14；DBSL 9/9 收盤下市
    reb = sched.loc[sched["signal"] == s, "rebal"].iloc[0]
    assert reb == pd.Timestamp("2021-09-07")
    expect = rz.at[pd.Timestamp("2021-09-09"), "DBSL"] / rzo.at[reb, "DBSL"] - 1
    assert np.isclose(hr.at[s, "DBSL"], expect)
    assert np.isnan(PP.holding_returns(rzo, sched, 1).at[s, "DBSL"])     # 沒給 rz 時是 NaN（舊行為）
    nxt = sched.loc[sched["signal"] > s, "signal"].iloc[0]
    assert np.isnan(hr.at[nxt, "DBSL"])                                  # 下市之後的持有期沒有報酬
