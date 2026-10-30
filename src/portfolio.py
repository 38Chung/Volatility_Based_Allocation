"""組合建構：選股、個股權重、權重上限、總曝險。

這裡產生的都是「目標」：只用到訊號日（含）以前的資料。換倉日開盤的成交限制（停牌、不交易區間）
在 backtest.py 處理。
"""
import numpy as np
import pandas as pd

import config as C
import volatility as V

SECTORS = sorted(set(C.SECTOR[t] for t in C.TICKERS))
UNCAPPED_SECTORS = ["Real Estate"]


def sector_index(tickers):
    return np.array([SECTORS.index(C.SECTOR[t]) for t in tickers])


# ---------------------------------------------------------------- 產業上限
def bench_sector_weights(mv_row, eligible, sec):
    """「市值加權」的產業權重：用訊號日當天的 MV，只算可交易股票。"""
    m = np.where(eligible & np.isfinite(mv_row) & (mv_row > 0), mv_row, 0.0)
    tot = m.sum()
    return np.array([m[sec == s].sum() / tot if tot > 0 else 0.0 for s in range(len(SECTORS))])


def sector_caps(bench, over):
    """產業上限 = 基準權重 + over；Real Estate 不設上限；over 為 None 時全部不設。"""
    caps = np.full(len(SECTORS), np.inf)
    if over is None:
        return caps
    for s, name in enumerate(SECTORS):
        if name not in UNCAPPED_SECTORS:
            caps[s] = bench[s] + over
    return caps


def apply_caps(w, sec, stock_cap, sec_caps, fixed=None, total=1.0, max_iter=100, tol=1e-12):
    """迭代處理上限：超過上限的部分砍掉、按比例分給沒有超限的股票，直到收斂。
    fixed 的權重不變（停牌、不交易區間），但會算進產業加總。sec_caps 是「占 total 的比例」。
    回傳 (w, status)：status = "ok" / "relaxed"（放寬產業上限才可行）/ "infeasible"。"""
    w = np.asarray(w, float).copy()
    fixed = np.zeros(len(w), bool) if fixed is None else fixed.copy()
    stock_cap = np.inf if stock_cap is None else stock_cap
    caps_abs = np.asarray(sec_caps, float) * total
    for attempt in range(25):
        x, ok = _caps_once(w, sec, stock_cap, caps_abs, fixed, total, max_iter, tol)
        if ok:
            return x, ("ok" if attempt == 0 else "relaxed")
        caps_abs = np.where(np.isfinite(caps_abs), caps_abs + 0.05 * total, np.inf)   # 做不到 → 放寬產業上限
    x, ok = _caps_once(w, sec, stock_cap, np.full_like(caps_abs, np.inf), fixed, total, max_iter, tol)
    return x, ("relaxed" if ok else "infeasible")


def _caps_once(w, sec, stock_cap, caps_abs, fixed, total, max_iter, tol):
    w = w.copy()
    free = ~fixed & (w > 0)
    room = total - w[fixed].sum()
    if free.sum() == 0:
        return w, True
    w[free] *= room / w[free].sum()
    at_cap = np.zeros(len(w), bool)
    sat = np.zeros(len(caps_abs), bool)
    for _ in range(max_iter):
        moved = False
        over = free & (w > stock_cap + tol)
        if over.any():
            excess = (w[over] - stock_cap).sum()
            w[over] = stock_cap
            at_cap |= over
            if not _distribute(w, excess, free & ~at_cap & ~sat[sec]):
                return w, False
            moved = True
        for s in range(len(caps_abs)):
            tot = w[sec == s].sum()
            if tot > caps_abs[s] + tol:
                fs = free & (sec == s)
                red = w[fs].sum()
                need = tot - caps_abs[s]
                if red < need - tol:
                    return w, False
                w[fs] *= (red - need) / red
                sat[s] = True
                if not _distribute(w, need, free & ~at_cap & ~sat[sec]):
                    return w, False
                moved = True
        if not moved:
            ok = np.all(w[free] <= stock_cap + 1e-10) and all(
                w[sec == s].sum() <= caps_abs[s] + 1e-10 for s in range(len(caps_abs)))
            return w, ok
    return w, False


def _distribute(w, amount, recv):
    base = w[recv].sum()
    if amount <= 0:
        return True
    if not recv.any() or base <= 0:
        return False
    w[recv] += amount * w[recv] / base
    return True


# ---------------------------------------------------------------- 選股
def select_with_buffer(score, eligible, held, K, buffer):
    """依分數排序取前 K；已持有且排名在前 K+buffer 的保留，空位由沒持有的股票依排名遞補。"""
    s = np.where(eligible & np.isfinite(score), score, -np.inf)
    order = np.argsort(-s, kind="stable")
    rank = np.empty(len(s), int); rank[order] = np.arange(len(s))
    valid = np.isfinite(s)
    keep = held & valid & (rank < K + buffer)
    if keep.sum() > K:                                  # 理論上不會發生（持有數 ≤ K），保險起見取排名最好的 K 檔
        keep_idx = np.where(keep)[0]
        keep[keep_idx[np.argsort(rank[keep_idx])[K:]]] = False
    sel = keep.copy()
    for i in order:
        if sel.sum() >= K:
            break
        if valid[i] and not sel[i]:
            sel[i] = True
    return sel


# ---------------------------------------------------------------- 個股權重
def target_weights(spec, sel, sig_row, score_row, mv_row, r, pos, sec, bench):
    """單一訊號日的相對目標權重（加總 1，已套用上限）。回傳 (w, info)。"""
    N = len(sel)
    w = np.zeros(N)
    idx = np.where(sel)[0]
    info = {"cap_status": "ok", "fallback": ""}
    if len(idx) == 0:
        return w, info
    kind = spec["weighting"]
    if kind == "ew":
        w[idx] = 1.0
    elif kind == "mv":
        w[idx] = np.where(mv_row[idx] > 0, mv_row[idx], 0)
    elif kind == "invvol":
        w[idx] = sig_row[idx] ** (-spec.get("gamma", 1.0))
    else:
        raise ValueError(kind)
    if w.sum() <= 0:
        return w, info
    w /= w.sum()
    if spec.get("stock_cap") is not None or spec.get("sector_over") is not None:
        w, st = apply_caps(w, sec, spec.get("stock_cap"), sector_caps(bench, spec.get("sector_over")))
        info["cap_status"] = st
    return w, info


# ---------------------------------------------------------------- 總曝險
def exposures(spec_exp, rel_w, r, sig_pos, sti_ret=None):
    """每個訊號日的目標曝險。spec_exp = {method, target, lookback, horizon?}。
    用新的相對權重回推組合過去 lookback 天的日報酬，預測組合波動 σ̂_p，曝險 = min(1, target / σ̂_p)。
    回傳 DataFrame(exposure, sigma_p, fallback)。"""
    n = len(sig_pos)
    out = {"exposure": np.full(n, np.nan), "sigma_p": np.full(n, np.nan), "fallback": np.zeros(n, bool)}
    target = spec_exp.get("target", C.VOL_TARGET)
    lb = spec_exp.get("lookback", C.GARCH_LOOKBACK)
    for k, pos in enumerate(sig_pos):
        if rel_w[k].sum() <= 0:
            continue
        y = V.portfolio_returns(r, pos, rel_w[k], lb)
        s, fb = V.forecast_port_sigma(y, spec_exp.get("method", "garch"), spec_exp.get("horizon", 5))
        out["sigma_p"][k], out["fallback"][k] = s, fb
        out["exposure"][k] = min(C.MAX_EXPOSURE, target / s) if np.isfinite(s) and s > 0 else C.MAX_EXPOSURE
    return pd.DataFrame(out)
