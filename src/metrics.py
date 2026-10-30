"""績效指標。r_f = 0，現金不計息。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

RF_NOTE = "r_f = 0，現金不計息"
RF_NOTE_EN = "r_f = 0, cash earns no interest"                       # 圖的註腳用


def max_drawdown(r):
    """日報酬序列的最大回撤（負數）。淨值從 1 開始，第一期的虧損也算進回撤。"""
    nav = pd.Series(np.r_[1.0, (1 + pd.Series(r).dropna()).cumprod().values])
    return float((nav / nav.cummax() - 1).min())


def drawdown_count(r, threshold=0.05):
    """從前高下跌超過 threshold、到回到前高為止算一次。"""
    nav = np.r_[1.0, (1 + pd.Series(r).dropna()).cumprod().values]
    peak, n, in_dd = nav[0], 0, False
    for v in nav[1:]:
        if v >= peak:
            peak, in_dd = v, False
        elif not in_dd and v / peak - 1 < -threshold:
            n, in_dd = n + 1, True
    return n


def ann_return(r):
    r = pd.Series(r).dropna()
    return float((1 + r).prod() ** (252 / len(r)) - 1) if len(r) else np.nan


def ann_vol(r):
    return float(pd.Series(r).dropna().std() * np.sqrt(252))


def var_es(r, q=0.95):
    r = pd.Series(r).dropna()
    v = np.percentile(r, 100 * (1 - q))
    return float(v), float(r[r <= v].mean())


def capm(r, rm):
    """r_p 對 r_STI 迴歸（r_f = 0），NW t 值。α 年化（×252）。"""
    df = pd.concat([pd.Series(r), pd.Series(rm)], axis=1).dropna()
    if len(df) < 30:
        return dict(alpha=np.nan, alpha_t=np.nan, beta=np.nan, beta_t=np.nan)
    lags = int(4 * (len(df) / 100) ** (2 / 9))
    res = sm.OLS(df.iloc[:, 0].values, sm.add_constant(df.iloc[:, 1].values)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    return dict(alpha=res.params[0] * 252, alpha_t=res.tvalues[0], beta=res.params[1], beta_t=res.tvalues[1])


def kupiec(n_exceed, n, p=0.05):
    """Kupiec POF 檢定。回傳 (LR, p 值)。"""
    if n == 0:
        return np.nan, np.nan
    x = n_exceed
    phat = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = ((n - x) * np.log(1 - phat) if phat < 1 else 0) + (x * np.log(phat) if x > 0 else 0)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def perf(daily, bench_ret, rm, trades=None, rebal=None, start=None, end=None, gross=False):
    """一個組合在一段期間的指標。daily: backtest 的 daily DataFrame。"""
    d = daily.loc[start:end]
    r = d["ret_gross"] if gross else d["ret"]
    out = {"start": d.index[0].date() if len(d) else None, "end": d.index[-1].date() if len(d) else None,
           "n_days": len(r), "ann_return": ann_return(r), "ann_vol": ann_vol(r)}
    out["sharpe"] = out["ann_return"] / out["ann_vol"] if out["ann_vol"] > 0 else np.nan
    out["max_drawdown"] = max_drawdown(r)
    out["n_drawdowns_5pct"] = drawdown_count(r)
    b = bench_ret.reindex(r.index)
    active = r - b
    out["excess_vs_B0"] = out["ann_return"] - ann_return(b)
    te = active.std() * np.sqrt(252)
    out["tracking_error"] = te
    out["info_ratio"] = active.mean() * 252 / te if te > 0 else np.nan
    out["VaR95"], out["ES95"] = var_es(r)
    out.update(capm(r, rm.reindex(r.index)))
    out["avg_cash"] = float(d["cash"].mean())
    if rebal is not None and len(rebal):
        rb = rebal.loc[start:end]
        out["avg_weekly_turnover"] = float(rb["one_way_turnover"].mean()) if len(rb) else np.nan
        out["avg_exposure"] = float(rb["stock_weight"].mean()) if len(rb) else np.nan
    else:
        out["avg_weekly_turnover"] = 0.0
        out["avg_exposure"] = 1.0
    if trades is not None and len(trades):
        tr = trades[(trades["buy_date"] >= pd.Timestamp(start or "1900")) & (trades["buy_date"] <= pd.Timestamp(end or "2100"))]
        out["n_trades"] = len(tr)
        out["win_rate"] = float((tr["pnl"] > 0).mean()) if len(tr) else np.nan
        out["avg_hold_days"] = float(tr["hold_days"].mean()) if len(tr) else np.nan
        out["avg_trade_pnl"] = float(tr["pnl"].mean()) if len(tr) else np.nan
    else:
        out.update(n_trades=0, win_rate=np.nan, avg_hold_days=np.nan, avg_trade_pnl=np.nan)
    return out


def stress(daily, windows):
    out = {}
    for name, (a, b) in windows.items():
        r = daily["ret"].loc[a:b]
        out[f"{name}_return"] = float((1 + r).prod() - 1) if len(r) else np.nan
        out[f"{name}_mdd"] = max_drawdown(r) if len(r) else np.nan
    return out


def brinson(port_w, bench_w, period_ret, sector):
    """每期 Brinson：配置 = Σ(w_p,s − w_b,s)(R_b,s − R_b)，選股 = Σ w_p,s (R_p,s − R_b,s)（含交互作用）。
    權重是期初（換倉後）股票部位內的相對權重。回傳每期 DataFrame。"""
    rows = []
    secs = sorted(set(sector))
    sec = np.array(sector)
    for d in port_w.index:
        wp, wb, R = port_w.loc[d].values, bench_w.loc[d].values, period_ret.loc[d].values
        ok = np.isfinite(R)
        wp = np.where(ok, wp, 0); wb = np.where(ok, wb, 0)
        if wp.sum() <= 0 or wb.sum() <= 0:
            continue
        wp, wb = wp / wp.sum(), wb / wb.sum()
        Rv = np.where(ok, R, 0)
        Rb = wb @ Rv
        alloc = sel = 0.0
        for s in secs:
            m = sec == s
            wps, wbs = wp[m].sum(), wb[m].sum()
            Rbs = (wb[m] @ Rv[m]) / wbs if wbs > 0 else Rb
            Rps = (wp[m] @ Rv[m]) / wps if wps > 0 else Rbs
            alloc += (wps - wbs) * (Rbs - Rb)
            sel += wps * (Rps - Rbs)
        rows.append({"date": d, "allocation": alloc, "selection": sel, "active": wp @ Rv - Rb})
    return pd.DataFrame(rows).set_index("date")
