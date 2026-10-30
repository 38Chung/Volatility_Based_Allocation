"""週頻單因子檢定（4.3）：Rank IC、Newey–West t、IC 衰減、因子相關矩陣。只用訓練期。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as C


def nw_tstat(x, lags=None):
    """序列均值的 Newey–West t 值；預設滯後 floor(4 (T/100)^(2/9))。"""
    x = pd.Series(x).dropna()
    if len(x) < 12:
        return np.nan
    if lags is None:
        lags = int(4 * (len(x) / 100) ** (2 / 9))
    res = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def rank_ic(factor, ret):
    """每期因子與報酬的 Spearman 相關；兩者都有值的股票少於 MIN_STOCKS 則該期不算。"""
    ic = {}
    for d in factor.index:
        both = pd.concat([factor.loc[d], ret.loc[d]], axis=1).dropna()
        if len(both) >= C.MIN_STOCKS:
            ic[d] = both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank())
    return pd.Series(ic, dtype=float)


def ic_summary(z, ret):
    """回傳 (摘要表, 每週 IC 表)。"""
    rows, series = {}, {}
    for k, f in z.items():
        ic = rank_ic(f, ret)
        series[k] = ic
        rows[k] = {"n_weeks": int(ic.notna().sum()), "IC_mean": ic.mean(), "IC_std": ic.std(),
                   "ICIR": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
                   "ICIR_annual": ic.mean() / ic.std() * np.sqrt(52) if ic.std() > 0 else np.nan,
                   "NW_t": nw_tstat(ic), "IC_pos_pct": (ic > 0).mean()}
    return pd.DataFrame(rows).T, pd.DataFrame(series)


def ic_decay(z, fwd_by_h):
    """fwd_by_h: {h: 訊號日 × 股票的 h 週累積報酬}。重疊樣本的 NW 滯後取 max(h, 預設)。"""
    out = {}
    for k, f in z.items():
        for h, ret in fwd_by_h.items():
            ic = rank_ic(f, ret.reindex(f.index))
            default = int(4 * (max(len(ic), 1) / 100) ** (2 / 9))
            out[(k, h)] = {"IC_mean": ic.mean(), "NW_t": nw_tstat(ic, lags=max(h, default)),
                           "n_weeks": int(ic.notna().sum())}
    df = pd.DataFrame(out).T
    df.index.names = ["factor", "h_weeks"]
    return df


def factor_corr(z):
    """因子間平均截面 Spearman 相關矩陣。"""
    keys = list(z)
    dates = z[keys[0]].index
    acc = np.zeros((len(keys), len(keys)))
    cnt = np.zeros_like(acc)
    for d in dates:
        m = pd.DataFrame({k: z[k].loc[d] for k in keys})
        c = m.corr(method="spearman", min_periods=C.MIN_STOCKS).values
        ok = ~np.isnan(c)
        acc[ok] += c[ok]
        cnt[ok] += 1
    return pd.DataFrame(acc / np.where(cnt > 0, cnt, np.nan), index=keys, columns=keys)
