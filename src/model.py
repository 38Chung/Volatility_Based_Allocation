"""因子合成：滾動 Lasso、Elastic Net、人工等權合成、單因子 fep。"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, enet_path, lasso_path

import config as C
from ic_test import nw_tstat

warnings.filterwarnings("ignore", category=ConvergenceWarning)

C_GRID = np.logspace(-3, 0, 30)[:-1]   # λ = c × λ_max，排除 c = 1（29 個格點，最大 ≈ 0.79）
EW_FACTORS = ["ep_ttm", "fep", "dy", "mom12_1", "rev5"]


def demeaned_target(hold_ret, tradable):
    """y = 持有期報酬 − 當期截面平均（只用可交易、報酬有值的股票）。(n_sig, N)"""
    y = np.where(tradable & np.isfinite(hold_ret), hold_ret, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # 最後一期沒有報酬
        return y - np.nanmean(y, axis=1, keepdims=True)


def eligible_periods(signal, y_end, j):
    """訊號日 j 可以用的訓練期數：y 的持有期結束（下次換倉日開盤）≤ 訊號日 t。"""
    return np.where((np.arange(len(signal)) < j) & (y_end <= signal[j]))[0]


def _stack(X, y, tradable, periods):
    rows, ys, grp = [], [], []
    for g, i in enumerate(periods):
        m = tradable[i] & np.isfinite(y[i])
        rows.append(X[i][m]); ys.append(y[i][m]); grp.append(np.full(m.sum(), g))
    return np.vstack(rows), np.concatenate(ys), np.concatenate(grp)


def _weekly_rank_ic(pred, y, grp):
    """pred: (n_rows, n_alpha)。回傳每個 α 的平均週 Rank IC（常數預測的週不計）。"""
    ics = np.full((grp.max() + 1, pred.shape[1]), np.nan)
    for g in range(grp.max() + 1):
        m = grp == g
        if m.sum() < C.MIN_STOCKS:
            continue
        ry = pd.Series(y[m]).rank().values
        P = pd.DataFrame(pred[m]).rank().values
        ry = ry - ry.mean(); P = P - P.mean(axis=0)
        den = np.sqrt((P ** 2).sum(axis=0) * (ry ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            ics[g] = np.where(den > 0, (P * ry[:, None]).sum(axis=0) / den, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(ics, axis=0)


# ---------------------------------------------------------------- 選 λ：λ 比例化與成本感知
def lambda_max(X, y, l1_ratio=1.0):
    """係數 ≥ 0、不估截距時的門檻：λ ≥ max_k X_k'y / (n · l1_ratio) 係數全為 0（sklearn 的目標函數尺度）。"""
    return float((X.T @ y).max() / (len(y) * l1_ratio))


def middle_of_ties(idx):
    """並列的格點（依 c 由小到大）取中間點：第 ⌊(n−1)/2⌋ 個，偶數個時取中間偏小。"""
    idx = np.sort(np.asarray(idx))
    return int(idx[(len(idx) - 1) // 2])


def sim_net_sharpe(scores, y, eligible, K=C.LAMBDA_SIM_K, buffer=C.LAMBDA_SIM_BUFFER, cost_bps=C.COST_BPS):
    """選 λ 用的簡化 S1：每週分數前 K 檔等權、排名緩衝 K + buffer，不在股票池（eligible 為 False）的持股賣出。
    scores, y, eligible：(週數, 股票)。週報酬 = 持股 y（超額報酬）的平均；成本 = cost_bps × Σ|Δw|（不計漂移；
    第一週從現金建倉不計成本）。回傳扣成本後週報酬的平均 / 標準差；不足 2 週或標準差為 0 時回傳 −inf。"""
    n, N = scores.shape
    w_old = None
    held = np.zeros(N, bool)
    net = []
    c = cost_bps / 1e4
    for i in range(n):
        ok = eligible[i] & np.isfinite(scores[i]) & np.isfinite(y[i])
        s = np.where(ok, scores[i], -np.inf)
        order = np.argsort(-s, kind="stable")
        rank = np.empty(N, int); rank[order] = np.arange(N)
        keep = held & ok & (rank < K + buffer)
        sel = keep.copy()
        for j in order:
            if sel.sum() >= K or not ok[j]:
                break
            sel[j] = True
        if not sel.any():
            continue
        w = sel / sel.sum()
        ret = float(y[i][sel].mean())
        cost = 0.0 if w_old is None else c * float(np.abs(w - w_old).sum())
        net.append(ret - cost)
        w_old, held = w, sel
    net = np.asarray(net)
    if len(net) < 2 or not net.std(ddof=1) > 0:
        return -np.inf
    return float(net.mean() / net.std(ddof=1))


def weekly_factor_ic(X, y, tradable):
    """每期、每個因子的截面 Rank IC（因子 vs y）。(n_sig, F)。
    X 的 NaN 已補成 0；中性化後的 z 值剛好等於 0 的機率是 0，所以 X == 0 視為缺值，跟階段 1 用補值前的版本一致。"""
    n, N, F = X.shape
    ic = np.full((n, F), np.nan)
    for i in range(n):
        base = tradable[i] & np.isfinite(y[i])
        for k in range(F):
            m = base & (X[i, :, k] != 0)
            if m.sum() >= C.MIN_STOCKS:
                ic[i, k] = pd.Series(X[i, m, k]).rank().corr(pd.Series(y[i][m]).rank())
    return ic


def screen_factors(ic, periods, t_min=C.SCREEN_T_MIN):
    """用訓練窗口內各期的週 Rank IC 算 Newey–West t 值，留下 t > t_min 的因子。回傳 (保留的欄位, 各因子 t 值)。"""
    t = np.array([nw_tstat(ic[periods, k]) for k in range(ic.shape[1])])
    return np.where(t > t_min)[0], t


def fit_at(X, y, tradable, y_end, t_date, kind="lasso", window=C.LASSO_WINDOW,
           min_window=C.LASSO_MIN_WINDOW, val_weeks=C.LAMBDA_VALID_WEEKS, ic=None, keep=None):
    """在任意訊號日 t_date 估計一次：只用持有期結束（換倉日開盤）≤ t_date 的週頻訓練資料。
    X, y, tradable 是週頻訓練資料（期數 × 股票）。資料不足回傳 None；否則回傳 (β, log dict)。
    先用同一個訓練窗口的週 Rank IC 篩選因子（NW t > SCREEN_T_MIN），只有通過的因子進 Lasso；
    ic 是 weekly_factor_ic 的結果（可以先算好傳進來）。
    keep：固定篩選的因子欄位（SCREEN_MODE = "static"）；給了就不做滾動篩選。"""
    el = np.where(y_end <= np.datetime64(t_date))[0]
    if len(el) < min_window:
        return None
    win = el if window == "expanding" else el[-window:]
    F = X.shape[2]
    if keep is not None:
        keep, tvals = np.asarray(keep, dtype=int), np.full(F, np.nan)
    else:
        if ic is None:
            ic = weekly_factor_ic(X, y, tradable)
        keep, tvals = screen_factors(ic, win)
    log = {"signal": pd.Timestamp(t_date), "n_periods": len(win), "first_y_start_idx": int(win[0]),
           "last_y_end": y_end[win[-1]], "n_screened": len(keep), "screen_idx": keep,
           **{f"t_{k}": tvals[k] for k in range(F)}}
    if len(keep) == 0:   # 沒有因子通過：不估計，分數用 19 個因子等權（score_from_beta 的備援）
        log.update({"alpha": np.nan, "val_ic": np.nan, "n_rows": 0, "val_first_idx": -1, "val_last_idx": -1,
                    "val_last_y_end": pd.NaT, "all_zero": True})
        return np.zeros(F), log
    Xs = X[:, :, keep]
    tr_p, va_p = win[:-val_weeks], win[-val_weeks:]
    Xtr, ytr, _ = _stack(Xs, y, tradable, tr_p)
    Xva, yva, gva = _stack(Xs, y, tradable, va_p)
    Xall, yall, _ = _stack(Xs, y, tradable, win)
    l1 = 1.0 if kind == "lasso" else 0.5
    path_fn = (lambda Xa, ya, al: lasso_path(Xa, ya, alphas=al, positive=True, max_iter=10000)) if kind == "lasso" \
        else (lambda Xa, ya, al: enet_path(Xa, ya, l1_ratio=0.5, alphas=al, positive=True, max_iter=10000))
    # λ = c × λ_max(前段)；準則 = 驗證段簡化 S1 扣成本後 Sharpe；並列取中間點；重估用 c × λ_max(整個窗口)
    lm_front, lm_full = lambda_max(Xtr, ytr, l1), lambda_max(Xall, yall, l1)
    extra = {"lmax_front": lm_front, "lmax_full": lm_full}
    if lm_front > 0:
        _, path, _ = path_fn(Xtr, ytr, C_GRID * lm_front)            # path: (k, n_c)，依 c 由小到大
        sc = np.einsum("pnk,kc->pnc", Xs[va_p], path)                 # 驗證段每週、每檔、每個 c 的分數
        yv = y[va_p]; el = tradable[va_p] & np.isfinite(yv)
        crit = np.array([sim_net_sharpe(sc[:, :, c], yv, el) for c in range(len(C_GRID))])
        ic = _weekly_rank_ic(Xva @ path, yva, gva)
        best = np.max(crit)
        cand = np.where(np.isfinite(crit) & np.isclose(crit, best, rtol=0, atol=1e-12))[0] \
            if np.isfinite(best) else np.arange(len(C_GRID))
        ci = middle_of_ties(cand)
        c_star = float(C_GRID[ci])
        extra.update({"c": c_star, "n_tied": len(cand), "val_net_sharpe": best if np.isfinite(best) else np.nan,
                      "val_ic_at_c": ic[ci]})
        best_ic = ic[ci]
    else:
        c_star, best_ic = np.nan, np.nan
        extra.update({"c": np.nan, "n_tied": 0, "val_net_sharpe": np.nan, "val_ic_at_c": np.nan})
    alpha = c_star * lm_full if (np.isfinite(c_star) and lm_full > 0) else np.nan
    best = best_ic
    beta = np.zeros(F)
    if np.isfinite(alpha) and alpha > 0:
        if kind == "lasso":
            m = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=10000)
        else:
            m = ElasticNet(alpha=alpha, l1_ratio=0.5, positive=True, fit_intercept=False, max_iter=10000)
        beta[keep] = m.fit(Xall, yall).coef_
    log.update({"alpha": alpha, "val_ic": best if np.isfinite(best) else np.nan, "n_rows": len(yall),
                "val_first_idx": int(va_p[0]), "val_last_idx": int(va_p[-1]),
                "val_last_y_end": y_end[va_p[-1]], "all_zero": not np.any(beta > 0),
                "n_nonzero": int((beta > 0).sum()), **extra})
    return beta, log


def score_from_beta(beta, X_t, tradable_t, fallback_idx=None):
    """分數 = X_t · β。β 全為 0 時改用等權平均：有通過篩選的因子就用那幾個，否則用全部因子。"""
    if np.any(beta > 0):
        s = X_t @ beta
    elif fallback_idx is not None and len(fallback_idx) > 0:
        s = X_t[:, fallback_idx].mean(axis=1)
    else:
        s = X_t.mean(axis=1)
    return np.where(tradable_t, s, np.nan)


def rolling_model(X, y, tradable, signal, y_end, kind="lasso", window=C.LASSO_WINDOW,
                  min_window=C.LASSO_MIN_WINDOW, val_weeks=C.LAMBDA_VALID_WEEKS, keep=None):
    """滾動估計：每個訊號日呼叫一次 fit_at。window = 整數或 "expanding"；keep = 固定篩選的因子欄位。
    回傳 (scores (n_sig, N)，NaN = 不出訊號；coef DataFrame；log DataFrame)。"""
    n_sig, N, F = X.shape
    scores = np.full((n_sig, N), np.nan)
    coefs, logs = {}, []
    ic = weekly_factor_ic(X, y, tradable) if keep is None else None
    for j in range(n_sig):
        res = fit_at(X, y, tradable, y_end, signal[j], kind, window, min_window, val_weeks, ic=ic, keep=keep)
        if res is None:
            continue
        beta, log = res
        scores[j] = score_from_beta(beta, X[j], tradable[j], log["screen_idx"])
        coefs[signal[j]] = beta
        logs.append(log)
    coef = pd.DataFrame(coefs).T
    return scores, coef, pd.DataFrame(logs)


def static_score(X, tradable, factor_names, use):
    idx = [factor_names.index(f) for f in use]
    s = X[:, :, idx].mean(axis=2)
    return np.where(tradable, s, np.nan)


def score_ic(scores, hold_ret, tradable, signal):
    """每週分數 vs 下一期持有期報酬的 Rank IC。"""
    out = {}
    for j in range(len(signal)):
        m = tradable[j] & np.isfinite(scores[j]) & np.isfinite(hold_ret[j])
        if m.sum() >= C.MIN_STOCKS:
            a = pd.Series(scores[j][m]).rank(); b = pd.Series(hold_ret[j][m]).rank()
            out[signal[j]] = a.corr(b)
    return pd.Series(out, dtype=float)
