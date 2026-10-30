"""19 個因子的每日值（交易日 × 股票）。

所有滾動窗口都以交易日為單位，不足窗口長度時為 NaN。
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import config as C
import data_loader as dl


def _returns(rz):
    return rz.pct_change(fill_method=None)


def _full(x, n):
    """窗口內 n 個觀測值都有值才算。"""
    return x.notna().astype(float).rolling(n, min_periods=n).sum() == n


def rolling_mdd(rz, n_ret=60):
    """窗口內（n_ret 個日報酬 = n_ret+1 個價格點）max(1 − RZ/滾動高點)。"""
    w = n_ret + 1
    vals = rz.values
    out = np.full(vals.shape, np.nan)
    if len(vals) >= w:
        win = sliding_window_view(vals, w, axis=0)          # (T-w+1, N, w)
        peak = np.maximum.accumulate(win, axis=2)
        dd = 1 - win / peak
        res = dd.max(axis=2)
        res[np.isnan(win).any(axis=2)] = np.nan
        out[w - 1:] = res
    return pd.DataFrame(out, index=rz.index, columns=rz.columns)


def rolling_capm(y, x, n, min_obs):
    """滾動 OLS y = α + β x + ε（含截距），只用 y、x 都有值的日子，至少 min_obs 天。
    回傳 (α, ε 的樣本標準差, 有效天數)；窗口不足 n 天或有效天數不足時為 NaN。"""
    both = y.notna() & x.notna()
    Y, X = y.where(both), x.where(both)
    cnt = both.astype(float).rolling(n, min_periods=n).sum()
    s = lambda v: v.rolling(n, min_periods=1).sum()
    mx, my = s(X) / cnt, s(Y) / cnt
    vxx = s(X * X) / cnt - mx ** 2
    vyy = s(Y * Y) / cnt - my ** 2
    vxy = s(X * Y) / cnt - mx * my
    beta = vxy / vxx
    alpha = my - beta * mx
    ssr = (vyy - vxy ** 2 / vxx).clip(lower=0) * cnt     # 殘差平方和
    sd = np.sqrt(ssr / (cnt - 1))
    ok = (cnt >= min_obs) & (vxx > 0)
    return alpha.where(ok), sd.where(ok & (sd > 0)), cnt.where(ok)


def value_months_ago(df, months):
    """每個日期 d 取 d − months 個曆月當天或之前最近一個交易日的值（沒有就是 NaN）。"""
    target = df.index - pd.DateOffset(months=months)
    pos = df.index.get_indexer(target, method="pad")      # 最近一個 ≤ target 的位置；沒有則 −1
    vals = df.values[np.clip(pos, 0, None)].copy()
    vals[pos < 0] = np.nan
    return pd.DataFrame(vals, index=df.index, columns=df.columns)


def dividends_ttm(panel, days):
    """過去 365 天 UDD 加總（新元）。在原始日期上滾動，再取交易日；資料第一年設為 NaN。"""
    udd = dl.field(panel, "UDD").fillna(0.0)
    for t, ccy in C.DIV_CCY.items():
        udd[t] = udd[t] * dl.rate_by_date(udd.index, ccy)
    ttm = udd.rolling("365D").sum()
    ttm.loc[ttm.index < ttm.index[0] + pd.DateOffset(years=1)] = np.nan
    # 資料期間內才上市的股票：上市後一年內股息窗口不完整，也設為 NaN
    rz = dl.field(panel, "RZ").reindex(days)
    first_day = rz.notna().any(axis=1).idxmax()
    for t in ttm.columns:
        listed = rz[t].first_valid_index()
        if listed is not None and listed > first_day:
            ttm.loc[ttm.index < listed + pd.DateOffset(years=1), t] = np.nan
    return ttm.reindex(days)


def compute_factors(panel, sti, days=None):
    """回傳 {因子代號: DataFrame(交易日 × 股票)}。"""
    if days is None:
        days = dl.trading_days(panel)
    p = panel.loc[days]
    rz, price, up, mv = (dl.field(p, k) for k in ["RZ", "P", "UP", "MV"])
    r = _returns(rz)
    r_sti = sti["RI"].reindex(days).pct_change(fill_method=None)

    f = {}
    f["mom20"] = rz / rz.shift(20) - 1
    f["mom60"] = rz / rz.shift(60) - 1
    f["mom12_1"] = rz.shift(21) / rz.shift(252) - 1
    f["rev5"] = -(rz / rz.shift(5) - 1)
    f["vol20"] = r.rolling(20, min_periods=20).std()
    f["down60"] = np.sqrt((r.clip(upper=0) ** 2).rolling(60, min_periods=60).mean())

    # beta60：只用 r_i 和 r_STI 都有值的日子，且 60 天都要有
    both = r.notna() & r_sti.notna().values[:, None]
    ri = r.where(both)
    rm = pd.DataFrame(np.repeat(r_sti.values[:, None], r.shape[1], axis=1), index=days, columns=r.columns).where(both)
    cov = (ri * rm).rolling(60, min_periods=60).mean() - ri.rolling(60, min_periods=60).mean() * rm.rolling(60, min_periods=60).mean()
    var = (rm ** 2).rolling(60, min_periods=60).mean() - rm.rolling(60, min_periods=60).mean() ** 2
    f["beta60"] = cov / var

    f["mdd60"] = rolling_mdd(rz, 60)

    uvo, nosh = dl.field(p, "UVO"), dl.field(p, "NOSH")
    f["turnover20"] = (uvo / nosh.where(nosh > 0)).rolling(20, min_periods=20).mean()

    va = dl.field(p, "VA")
    illiq = (r.abs() / va.where(va > 0)) * 1e6           # VA ≤ 0 的日子不納入
    hist20 = _full(r, 20)                                # 要有 20 天的報酬歷史
    f["amihud20"] = illiq.rolling(20, min_periods=C.TRADABLE_MIN_DAYS).mean().where(hist20)

    f["size"] = np.log(mv.where(mv > 0))

    f["dy"] = dividends_ttm(panel, days) / up.where(up > 0)

    eps = dl.field(p, "EPS1FD12")
    eps_chg = eps - value_months_ago(eps, 3)             # 先相減、再換匯
    eps_sgd, chg_sgd = eps.copy(), eps_chg.copy()
    for t, ccy in C.EPS_CCY.items():
        rate = dl.rate_by_date(days, ccy)
        eps_sgd[t] = eps[t] * rate
        chg_sgd[t] = eps_chg[t] * rate
    f["fep"] = eps_sgd / price.where(price > 0)
    f["rev"] = chg_sgd / price.where(price > 0)

    # ---- ep_ttm、ivol60、max20、overnight20、intraday20
    pe, eps_ttm = dl.field(p, "PE"), dl.field(p, "EPS")
    pe5, eps5, p5 = pe.shift(C.EP_TTM_LAG), eps_ttm.shift(C.EP_TTM_LAG), price.shift(C.EP_TTM_LAG)
    ep = (1 / pe5.where(pe5 > 0)) * p5 / price.where(price > 0)
    loss = ~(pe5 > 0) & eps5.notna()                      # PE 空值或 ≤ 0、EPS 有值 → 虧損，設為 0
    f["ep_ttm"] = ep.mask(loss, 0.0)

    rm_df = pd.DataFrame(np.repeat(r_sti.values[:, None], r.shape[1], axis=1), index=days, columns=r.columns)
    n_iv = 60
    _, sd_eps, _ = rolling_capm(r, rm_df, n_iv, int(np.ceil(C.REG_MIN_FRAC * n_iv)))
    f["ivol60"] = sd_eps

    f["max20"] = r.rolling(20, min_periods=20).max()

    rzo = rz * dl.field(p, "PO") / price.where(price > 0)
    f["overnight20"] = np.log(rzo / rz.shift(1)).rolling(20, min_periods=20).sum()
    f["intraday20"] = np.log(rz / rzo).rolling(20, min_periods=20).sum()

    if C.USE_BM:
        f["bm"] = _bm(p, mv)

    listed = rz.notna() & mv.notna() & (mv > 0)
    out = {}
    for k in C.FACTORS:
        v = f[k].replace([np.inf, -np.inf], np.nan).where(listed)
        out[k] = v[C.TICKERS]
    return out


def _shift_months(df, months):
    shifted = df.copy()
    shifted.index = shifted.index + pd.DateOffset(months=months)
    shifted = shifted[~shifted.index.duplicated(keep="last")]
    return shifted.reindex(df.index, method="ffill")


def _bm(p, mv):
    """選用因子 bm（預設關閉），照組員 fundamental_factors.py 的寫法。"""
    ce_raw = dl.field(p, "WC03501")
    fx = mv / (dl.field(p, "MTBV") * ce_raw)
    fx = fx.where(fx > 0).rolling(63, min_periods=5).median().ffill()
    for t in [t for t in ["EMPE", "IHHH"] if t in C.TICKERS]:
        fx[t] = dl.rate_by_date(fx.index, C.EPS_CCY[t]) / 1000
    fx = _shift_months(fx, 3)
    ce = _shift_months(ce_raw, 3)
    # FY2025 缺值往前補，最多補到財報日期（推遲後）後 24 個月
    ce = ce.ffill(limit=None)
    stamp = pd.DataFrame(np.where(_shift_months(ce_raw, 3).notna(), 1, np.nan), index=ce.index, columns=ce.columns)
    last_obs = stamp.mul(np.arange(len(ce))[:, None]).ffill()
    age_days = (pd.Series(ce.index).values[:, None] - ce.index.values[last_obs.fillna(0).astype(int).values])
    too_old = age_days > np.timedelta64(730, "D")
    ce = ce.mask(too_old | last_obs.isna().values)
    return (ce * fx / mv).where(ce > 0)


def save_factors(fac, path):
    long = pd.concat({k: v.stack(future_stack=True) for k, v in fac.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    long.to_parquet(path)
