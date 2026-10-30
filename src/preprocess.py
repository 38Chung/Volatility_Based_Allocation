"""週頻日曆（4.1）與截面預處理（4.2）。"""
import numpy as np
import pandas as pd

import config as C
import data_loader as dl


# ---------------------------------------------------------------- 4.1 週頻日曆
def weekly_schedule(days, weekday=C.REBALANCE_WEEKDAY):
    """每週從 weekday（預設週二）起的第一個交易日換倉；訊號日 = 換倉日前一個交易日。
    整週 weekday..週五都不是交易日則該週不換倉。回傳 DataFrame(signal, rebal, next_rebal)。"""
    days = pd.DatetimeIndex(days).sort_values()
    cand = days[(days.day_of_week >= weekday) & (days.day_of_week <= 4)]
    week = cand.to_period("W-SUN")                      # 週一到週日為一週
    rebal = pd.Series(cand, index=cand).groupby(week).min().values
    rebal = pd.DatetimeIndex(rebal)
    pos = days.get_indexer(rebal)
    keep = pos > 0                                      # 需要有前一個交易日當訊號日
    rebal, pos = rebal[keep], pos[keep]
    sched = pd.DataFrame({"signal": days[pos - 1], "rebal": rebal})
    sched["next_rebal"] = sched["rebal"].shift(-1)
    return sched.reset_index(drop=True)


def rz_open(panel, days):
    """含息開盤指數 RZ_open = RZ × PO / P。"""
    p = panel.loc[days]
    rz, po, px = dl.field(p, "RZ"), dl.field(p, "PO"), dl.field(p, "P")
    return rz * po / px.where(px > 0)


def holding_returns(rzo, sched, horizon=1, rz=None):
    """以訊號日為索引：RZ_open(第 i+h 個換倉日) / RZ_open(第 i 個換倉日) − 1。
    rz（收盤 RZ，日期 × 股票）有給時，持有期內下市的股票（config.DELISTED，換倉日 i ≤ L_i < 換倉日 i+h）
    報酬算到 L_i 收盤：RZ(L_i) / RZ_open(換倉日 i) − 1，之後視為現金（報酬 0）。"""
    reb = pd.DatetimeIndex(sched["rebal"])
    start = rzo.reindex(reb).values
    end = rzo.reindex(reb).shift(-horizon).values.copy()
    if rz is not None:
        end_date = pd.Series(reb).shift(-horizon)
        for t, d in C.DELISTED.items():
            L = pd.Timestamp(d)
            if t not in rzo.columns or L not in rz.index:              # 資料（截斷後）在 L_i 之前就結束
                continue
            j = rzo.columns.get_loc(t)
            hit = ((reb <= L) & (end_date > L)).values & np.isnan(end[:, j])
            end[hit, j] = rz.at[L, t]
    return pd.DataFrame(end / start - 1, index=pd.DatetimeIndex(sched["signal"]), columns=rzo.columns)


# ---------------------------------------------------------------- 4.2 截面預處理
def winsorize_mad(x, k=C.MAD_K):
    med = x.median()
    mad = 1.4826 * (x - med).abs().median()
    if not mad > 0:
        return x
    return x.clip(med - k * mad, med + k * mad)


def standardize(x):
    sd = x.std()
    if not sd > 0:
        return x * np.nan
    return (x - x.mean()) / sd


def preprocess_cross_section(raw, sector, log_mv, neutralize_size=True):
    """單一日期、單一因子。raw 只含可交易股票。回傳 (z, status)；z 未乘方向、未補 0。"""
    x = raw.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < C.MIN_STOCKS:
        return raw * np.nan, "fewer_than_10"
    x = standardize(winsorize_mad(x))
    if x.isna().all():
        return raw * np.nan, "zero_dispersion"
    X = pd.get_dummies(sector.reindex(x.index), dtype=float)   # 產業虛擬變數（含截距）
    if neutralize_size:
        X["log_mv"] = standardize(log_mv.reindex(x.index))
    if X.isna().any().any() or len(x) < X.shape[1] + 5 or np.linalg.matrix_rank(X.values) < X.shape[1]:
        return raw * np.nan, "neutralization_unavailable"
    beta = np.linalg.lstsq(X.values, x.values, rcond=None)[0]
    z = standardize(pd.Series(x.values - X.values @ beta, index=x.index))
    if z.isna().all():
        return raw * np.nan, "zero_dispersion_after_neutralization"
    return z.reindex(raw.index), "ok"


def preprocess_all(fac, tradable, mv, signal_dates, factors=None):
    """對每個訊號日、每個因子做 4.2 的步驟 1–5。
    回傳 (z: {因子: 訊號日 × 股票，已乘方向，未補 0}, status DataFrame)。"""
    sector = pd.Series(C.SECTOR)
    signal_dates = pd.DatetimeIndex(signal_dates)
    trad = tradable.reindex(signal_dates).fillna(False).astype(bool)
    log_mv = np.log(mv.reindex(signal_dates).where(mv.reindex(signal_dates) > 0))
    out, status = {}, []
    for k in (factors or C.FACTORS):
        raw_all = fac[k].reindex(signal_dates)
        rows = np.full(raw_all.shape, np.nan)
        for i, d in enumerate(signal_dates):
            names = trad.columns[trad.loc[d].values]
            z, st = preprocess_cross_section(raw_all.loc[d, names], sector, log_mv.loc[d],
                                             neutralize_size=(k != "size"))
            rows[i, raw_all.columns.get_indexer(names)] = z.values
            status.append({"factor": k, "date": d, "status": st, "n_valid": int(z.notna().sum())})
        out[k] = pd.DataFrame(rows, index=signal_dates, columns=raw_all.columns) * C.FACTOR_SIGN[k]
    return out, pd.DataFrame(status)


def model_input(z):
    """模型輸入：NaN 補 0（只在這裡補；IC 檢定用補值前的版本）。"""
    return {k: v.fillna(0.0) for k, v in z.items()}


def z_to_long(z):
    long = pd.concat({k: v.stack(future_stack=True) for k, v in z.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    return long
