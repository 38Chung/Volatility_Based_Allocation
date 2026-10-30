"""波動率估計：個股 σ̂ᵢ（6.3、E1）與組合 σ̂_p（6.4、V 系列）。

所有函式在訊號日 t 只用到 t（含）以前的日報酬；回傳的都是「日」波動（未年化），
組合波動另外提供年化版本。
"""
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

import config as C

warnings.filterwarnings("ignore", module="arch")

STOCK_METHODS = ["sd20", "sd60", "ewma94", "ewma97", "park20", "garch"]
PORT_METHODS = ["garch", "ewma"]   # 組合 σ̂_p：GARCH(1,1)-t；估計失敗時用 EWMA 備援


# ---------------------------------------------------------------- 個股 σ̂ᵢ
def _window(a, pos, n):
    """a 的第 pos−n+1 … pos 列（含 pos）。複製成 Fortran 排列（每檔股票的時間序列在記憶體中連續），
    沿時間軸加總時不受陣列記憶體位置影響，結果逐位元可重現（前視偏誤測試要求完全相同）。"""
    return np.asfortranarray(a[max(0, pos - n + 1):pos + 1])


def _ewma_sigma(win, lam):
    """win: (n, N) 日報酬，最後一列是最新。缺值不納入，權重重新正規化。零均值。"""
    n = win.shape[0]
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)[:, None]
    ok = ~np.isnan(win)
    num = np.where(ok, w * win ** 2, 0).sum(axis=0)
    den = np.where(ok, w, 0).sum(axis=0)
    out = np.sqrt(num / np.where(den > 0, den, np.nan))
    out[ok.sum(axis=0) < 20] = np.nan
    return out


def stock_sigma(r, sig_pos, method, ph=None, pl=None, traded=None, garch_refit=4, lookback=500):
    """每個訊號日（r 的列位置 sig_pos）每檔股票的日波動 σ̂ᵢ。r: (T, N) ndarray。回傳 (n_sig, N)。"""
    out = np.full((len(sig_pos), r.shape[1]), np.nan)
    if method == "garch":
        return _stock_garch(r, sig_pos, garch_refit, lookback)
    if method == "park20":
        hl = np.log(ph / pl) ** 2 / (4 * np.log(2))
        hl[~traded | ~np.isfinite(hl)] = np.nan
    for k, pos in enumerate(sig_pos):
        if method in ("sd20", "sd60"):
            n = 20 if method == "sd20" else 60
            win = _window(r, pos, n)
            cnt = np.sum(~np.isnan(win), axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                sd = np.nanstd(win, axis=0, ddof=1) if win.shape[0] > 1 else np.full(r.shape[1], np.nan)
            sd[cnt < int(0.8 * n)] = np.nan
            out[k] = sd
        elif method in ("ewma94", "ewma97"):
            lam = 0.94 if method == "ewma94" else 0.97
            out[k] = _ewma_sigma(_window(r, pos, 60), lam)
        elif method == "park20":
            win = _window(hl, pos, 20)
            cnt = np.sum(~np.isnan(win), axis=0)
            with np.errstate(invalid="ignore"):
                v = np.nanmean(win, axis=0)
            v[cnt < 10] = np.nan
            out[k] = np.sqrt(v)
        else:
            raise ValueError(method)
    return out


def _fit_garch(y, o=0):
    """y 是 ×100 的日報酬。回傳 (result, ok)。"""
    try:
        res = arch_model(y, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="t").fit(
            disp="off", show_warning=False)
        ok = res.convergence_flag == 0 and np.all(np.isfinite(res.params.values))
        return res, ok
    except Exception:
        return None, False


def _stock_garch(r, sig_pos, refit, lookback):
    """個股 GARCH(1,1)-t：每 refit 個訊號日重新估計參數，其他週用固定參數、用到訊號日的資料預測未來 5 天。
    估計失敗、或有效樣本 < 250 天時，改用 EWMA 0.94。"""
    n_sig, N = len(sig_pos), r.shape[1]
    out = np.full((n_sig, N), np.nan)
    params = [None] * N
    for k, pos in enumerate(sig_pos):
        win = _window(r, pos, lookback)
        fallback = _ewma_sigma(_window(r, pos, 60), 0.94)
        for i in range(N):
            y = win[:, i]
            y = y[~np.isnan(y)] * 100
            if len(y) < 250:
                out[k, i] = fallback[i]
                params[i] = None
                continue
            am = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
            if params[i] is None or k % refit == 0:
                res, ok = _fit_garch(y)
                params[i] = res.params if ok else None
            if params[i] is None:
                out[k, i] = fallback[i]
                continue
            try:
                var = am.fix(params[i]).forecast(horizon=5, reindex=False).variance.values[-1]
                s = np.sqrt(var.mean()) / 100
                out[k, i] = s if np.isfinite(s) and s > 0 else fallback[i]
            except Exception:
                out[k, i] = fallback[i]
    return out


def floor_sigma(sig, eligible, pct=C.SIGMA_FLOOR_PCT):
    """σ̂ᵢ 下限 = 當期所有（eligible）股票 σ̂ 的第 pct 百分位數。"""
    out = sig.copy()
    for k in range(sig.shape[0]):
        v = sig[k, eligible[k] & np.isfinite(sig[k])]
        if len(v):
            out[k] = np.maximum(sig[k], np.percentile(v, pct))
    return out


# ---------------------------------------------------------------- 組合 σ̂_p
def portfolio_returns(r, pos, w, lookback):
    """用權重 w（加總 1）回推訊號日 pos（含）以前 lookback 天的組合日報酬；缺值股票用剩下的重新正規化。"""
    win = _window(r, pos, lookback)
    ok = ~np.isnan(win) & (w[None, :] > 0)
    ww = np.where(ok, w[None, :], 0)
    den = ww.sum(axis=1)
    rp = np.where(ok, win, 0) @ w / np.where(den > 0, den, np.nan)
    return rp[np.isfinite(rp)]


def ewma_port(y, lam=C.EWMA_LAMBDA):
    """EWMA 日變異數（零均值），用整段 y 遞迴；起始值 = 前 20 天樣本變異數。"""
    if len(y) < 20:
        return np.nan
    v = np.var(y[:20])
    for x in y[20:]:
        v = lam * v + (1 - lam) * x * x
    return np.sqrt(v)


def forecast_port_sigma(y, method, horizon=5):
    """y：組合日報酬（小數）。horizon：GARCH 預測天數（週頻 5；月頻對照實驗 21）。回傳 (年化 σ̂_p, 是否用了 EWMA 備援)。"""
    if len(y) < 60:
        return np.nan, False
    if method == "garch":
        res, ok = _fit_garch(y * 100)
        if ok:
            var = res.forecast(horizon=horizon, reindex=False).variance.values[-1]
            s = np.sqrt(var.mean() * 252) / 100
            if np.isfinite(s) and s > 0:
                return s, False
        return ewma_port(y) * np.sqrt(252), True
    if method == "ewma":
        return ewma_port(y) * np.sqrt(252), False
    raise ValueError(method)
