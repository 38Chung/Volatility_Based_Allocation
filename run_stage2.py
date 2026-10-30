"""階段 2：模型與回測（只用到 2026-09-30 的資料）。

資料截斷 → 19 個因子與 z 值（用截斷後的資料重算）→ 滾動 Lasso 與對照模型 → 8 組組合回測
→ 波動率實驗與 K、SECTOR_OVER → 只用驗證期選參數 → frozen_params.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as BT
import config as C
import data_loader as dl
import factors as F
import ic_test as IC
import metrics as M
import model as MD
import plots as PL
import portfolio as P
import preprocess as PP
import volatility as V

PERIODS = {"full": (None, C.FREEZE_DATE), "train": (None, C.TRAIN_END), "valid": (C.VALID_START, C.VALID_END)}
STRESS = {"stress_2015": ("2015-06-01", "2016-02-29"), "stress_2020": ("2020-02-01", "2020-04-30"),
          "stress_2022": ("2022-01-01", "2022-12-31")}
SEL_MARGIN = 0.05            # 驗證期 Sharpe 比基準高不到 0.05 → 保留基準

BASE = dict(universe="select", weighting="invvol", gamma=1.0, sigma="ewma94", K=C.K, buffer=C.RANK_BUFFER,
            stock_cap=C.MAX_WEIGHT, sector_over=C.SECTOR_OVER, band=C.NO_TRADE_BAND, exposure=None,
            model="lasso260", cost_bps=C.COST_BPS)
EXP_BASE = dict(method="garch", target=C.VOL_TARGET, lookback=C.GARCH_LOOKBACK)
MANIFEST = []


def out(kind, name):
    d = {"data": C.DATA_DIR, "table": C.TABLE_DIR, "fig": C.FIG_DIR, "log": C.LOG_DIR}[kind]
    path = os.path.join(d, name)
    MANIFEST.append(os.path.relpath(path, C.ROOT))
    return path


def spec(**kw):
    s = dict(BASE); s.update(kw)
    return s


def with_exp(s, **kw):
    e = dict(EXP_BASE); e.update(kw)
    s = dict(s); s["exposure"] = e
    return s


# ====================================================================== 資料與特徵
def load_truncated(freeze=C.FREEZE_DATE):
    panel, sti = dl.load_panel(rebuild=False)
    panel = panel.loc[:freeze]
    sti = sti.loc[:freeze]
    return panel, sti


class Context:
    """截斷後資料的所有中間結果（陣列都是 訊號日 × 股票）。"""

    def __init__(self, panel, sti, verbose=True, membership=C.MEMBERSHIP_MASK, exclude=(), screen=None):
        t0 = time.time()
        self.panel, self.sti = panel, sti
        self.days = dl.trading_days(panel)
        self.tickers = C.TICKERS
        self.N = len(self.tickers)
        trad_daily = dl.tradable_mask(panel, self.days, membership=membership, exclude=exclude)
        self.trad_daily = trad_daily
        sched = PP.weekly_schedule(self.days)
        self.sched = sched[sched["signal"] >= C.START_SIGNAL].reset_index(drop=True)
        self.signal = pd.DatetimeIndex(self.sched["signal"])
        self.rebal = pd.DatetimeIndex(self.sched["rebal"])
        self.y_end = pd.DatetimeIndex(self.sched["next_rebal"]).values      # 最後一期為 NaT
        self.sig_pos = self.days.get_indexer(self.signal)
        self.tradable = trad_daily.reindex(self.signal).fillna(False).astype(bool).values
        self.fac = F.compute_factors(panel, sti, self.days)
        mv_daily = dl.field(panel.loc[self.days], "MV")
        z, self.prep_status = PP.preprocess_all(self.fac, trad_daily, mv_daily, self.signal)
        self.z = z
        self.factor_names = list(C.FACTORS)
        self.X = np.stack([z[k].fillna(0.0).values for k in self.factor_names], axis=2)
        rzo = PP.rz_open(panel, self.days)
        self.hold_ret = PP.holding_returns(rzo, self.sched, 1, rz=dl.field(panel.loc[self.days], "RZ")).values
        self.y = MD.demeaned_target(self.hold_ret, self.tradable)
        # 進 Lasso 前的 IC 篩選。static：用訓練期的週 IC 表固定一組因子；screen 可以直接指定（階段 3 用凍結的清單）
        self.screen_table = None
        if screen is not None:
            self.screen = list(screen)
        elif C.SCREEN_MODE == "static":
            self.screen, self.screen_table = self._static_screen()
        elif C.SCREEN_MODE is None:
            self.screen = list(self.factor_names)
        else:
            self.screen = None                                                  # rolling：每期在 fit_at 裡篩選
        self.mv = mv_daily.reindex(self.signal).values
        self.mkt = BT.Market.from_panel(panel, self.days)
        rz = dl.field(panel.loc[self.days], "RZ").values
        with np.errstate(invalid="ignore", divide="ignore"):
            self.r = rz / np.vstack([np.full((1, self.N), np.nan), rz[:-1]]) - 1     # 日報酬，上市前 NaN
        p = panel.loc[self.days]
        self.ph, self.pl = dl.field(p, "PH").values, dl.field(p, "PL").values
        self.traded = np.nan_to_num(dl.field(p, "VO").values, nan=0.0) > 0
        self.sti_ret = sti["RI"].reindex(self.days).pct_change(fill_method=None).values
        self.sec = P.sector_index(self.tickers)
        self.scores, self.model_logs, self.model_coefs = {}, {}, {}
        self.sigma_cache = {}
        self.j0 = None
        if verbose:
            print(f"  特徵完成（{time.time() - t0:.0f}s）：交易日 {len(self.days)}，訊號日 {len(self.signal)}，"
                  f"最後交易日 {self.days[-1].date()}")

    def _static_screen(self):
        """訓練期（訊號日 ≤ TRAIN_END）每週 Rank IC 的 NW t 值 > SCREEN_T_MIN 的因子（跟階段 1 的 IC 表同一個算法）。"""
        train = self.signal <= pd.Timestamp(C.TRAIN_END)
        ret = pd.DataFrame(self.hold_ret, index=self.signal, columns=self.tickers)[train]
        summ, _ = IC.ic_summary({k: self.z[k][train] for k in self.factor_names}, ret)
        keep = [k for k in self.factor_names if summ.loc[k, "NW_t"] > C.SCREEN_T_MIN]
        return keep, summ

    # ---------------- 模型分數
    def fit_models(self, which=("lasso260", "enet", "lasso156", "lasso_exp")):
        for name in which:
            if name in self.scores:
                continue
            t0 = time.time()
            kind = "enet" if name == "enet" else "lasso"
            window = {"lasso260": C.LASSO_WINDOW, "enet": C.LASSO_WINDOW, "lasso156": 156,
                      "lasso_exp": "expanding"}[name]
            keep = None if self.screen is None else [self.factor_names.index(f) for f in self.screen]
            s, coef, log = MD.rolling_model(self.X, self.y, self.tradable, self.signal.values, self.y_end,
                                            kind=kind, window=window, keep=keep)
            self.scores[name], self.model_coefs[name], self.model_logs[name] = s, coef, log
            print(f"  {name}：{len(log)} 期（{time.time() - t0:.0f}s），β 全為 0 的期數 {int(log['all_zero'].sum())}，"
                  f"篩選後平均 {log['n_screened'].mean():.1f} 個因子，沒有因子通過的期數 {int((log['n_screened'] == 0).sum())}")
        self.scores["ew5"] = MD.static_score(self.X, self.tradable, self.factor_names, MD.EW_FACTORS)
        self.scores["fep"] = MD.static_score(self.X, self.tradable, self.factor_names, ["fep"])
        if self.j0 is None:
            self.j0 = int(np.where(np.isfinite(self.scores["lasso260"]).any(axis=1))[0][0])

    # ---------------- σ̂ᵢ（只算 j0 以後的訊號日）
    def sigma(self, method):
        if method not in self.sigma_cache:
            t0 = time.time()
            pos = self.sig_pos[self.j0:]
            raw = V.stock_sigma(self.r, pos, method, ph=self.ph, pl=self.pl, traded=self.traded)
            full = np.full((len(self.signal), self.N), np.nan)
            full[self.j0:] = raw
            elig = self.tradable & np.isfinite(full)
            self.sigma_cache[method] = (full, V.floor_sigma(full, elig))
            print(f"  σ̂ {method}（{time.time() - t0:.0f}s）")
        return self.sigma_cache[method][1]


# ====================================================================== 組合
def build_targets(ctx, s, end_j=None):
    """回傳 dict(rel (n, N), caps (n, n_sec), sigma (n, N), exposure df, info list)，n = 從 j0 起的訊號日數。"""
    j0 = ctx.j0
    js = range(j0, len(ctx.signal) if end_j is None else end_j)
    sig_all = ctx.sigma(s["sigma"])
    scores = ctx.scores.get(s["model"]) if (s["universe"] == "select") else None
    held = np.zeros(ctx.N, bool)
    rel, caps, sigs, infos = [], [], [], []
    for j in js:
        sig = sig_all[j]
        elig = ctx.tradable[j] & np.isfinite(sig)
        sc = scores[j] if scores is not None else np.full(ctx.N, np.nan)
        if s["universe"] == "select":
            sel = P.select_with_buffer(sc, elig, held, s["K"], s["buffer"])
            held = sel
        else:
            sel = elig
        bench = P.bench_sector_weights(ctx.mv[j], ctx.tradable[j], ctx.sec)
        w, info = P.target_weights(s, sel, sig, sc, ctx.mv[j], ctx.r, ctx.sig_pos[j], ctx.sec, bench)
        rel.append(w); caps.append(P.sector_caps(bench, s["sector_over"])); sigs.append(sig); infos.append(info)
    rel = np.array(rel)
    exp = None
    if s["exposure"] is not None:
        exp = P.exposures(s["exposure"], rel, ctx.r, ctx.sig_pos[j0:j0 + len(rel)], ctx.sti_ret)
    # 選股組：賣不掉的持股要少買幾檔新股時，依當期分數決定（回測引擎用）
    priority = np.array([scores[j] for j in js]) if s["universe"] == "select" else None
    return dict(rel=rel, caps=np.array(caps), sigma=np.array(sigs), exposure=exp, info=infos, priority=priority)


def run_strategy(ctx, s, tg=None):
    tg = tg or build_targets(ctx, s)
    n = len(tg["rel"])
    rebal = ctx.rebal[ctx.j0:ctx.j0 + n]
    expo = tg["exposure"]["exposure"].values if tg["exposure"] is not None else None
    res = BT.run(ctx.mkt, rebal, tg["rel"], exposure=expo, cost_bps=s["cost_bps"], band=s["band"],
                 stock_cap=s["stock_cap"], sector=ctx.sec,
                 sector_caps_rel=tg["caps"] if s["sector_over"] is not None else None,
                 end_date=ctx.days[-1], sigma=tg["sigma"], priority=tg["priority"])
    res["targets"] = tg
    res["spec"] = s
    return res


def sti_daily(ctx):
    return pd.Series(ctx.sti_ret, index=ctx.days)


def evaluate(ctx, res, name, b0, gross_too=True):
    rows = []
    rm = sti_daily(ctx)
    for per, (a, b) in PERIODS.items():
        for g in ([False, True] if gross_too else [False]):
            m = M.perf(res["daily"], b0, rm, res["trades"], res["rebal"], a, b, gross=g)
            if name == "B0":                                # 基準本身：相對 B0、對 STI 的迴歸沒有意義
                m.update(alpha=np.nan, alpha_t=np.nan, beta=np.nan, beta_t=np.nan, info_ratio=np.nan,
                         tracking_error=np.nan, excess_vs_B0=np.nan)
            m.update(portfolio=name, period=per, cost="gross" if g else "net")
            rows.append(m)
    return rows


# ====================================================================== 實驗
def vol_extras(ctx, res, target):
    """V 系列：每週事後實現波動 vs 目標的 MAE、95% VaR 超出次數與 Kupiec（依期間）。"""
    exp = res["targets"]["exposure"]
    daily = res["daily"]["ret"]
    reb = res["rebal"].index
    out = {}
    for per, (a, b) in PERIODS.items():
        errs, n_days, n_exc = [], 0, 0
        for k in range(len(reb)):
            if not (pd.Timestamp(a or "1900") <= reb[k] <= pd.Timestamp(b)):
                continue
            end = reb[k + 1] if k + 1 < len(reb) else daily.index[-1] + pd.Timedelta(days=1)
            r = daily.loc[reb[k]:end - pd.Timedelta(days=1)]
            if len(r) >= 3:
                errs.append(abs(r.std() * np.sqrt(252) - target))
            sp = exp["sigma_p"].iloc[k]
            if np.isfinite(sp):
                var = 1.645 * exp["exposure"].iloc[k] * sp / np.sqrt(252)
                n_days += len(r); n_exc += int((r < -var).sum())
        lr, pv = M.kupiec(n_exc, n_days)
        out[per] = {"vol_mae": float(np.mean(errs)) if errs else np.nan, "var_days": n_days, "var_exceed": n_exc,
                    "var_exceed_rate": n_exc / n_days if n_days else np.nan, "kupiec_LR": lr, "kupiec_p": pv,
                    "ewma_fallbacks": int(exp["fallback"].sum())}
    return out


def experiment_rows(ctx, b0, series, option, base_name, s, is_base=False, extras=None):
    res = run_strategy(ctx, s)
    rows = []
    st = M.stress(res["daily"], STRESS)
    ve = vol_extras(ctx, res, s["exposure"]["target"]) if (extras and s["exposure"]) else None
    for per, (a, b) in PERIODS.items():
        m = M.perf(res["daily"], b0, sti_daily(ctx), res["trades"], res["rebal"], a, b)
        row = {"series": series, "option": option, "portfolio": base_name, "period": per, "is_base": is_base,
               "ann_return": m["ann_return"], "ann_vol": m["ann_vol"], "sharpe": m["sharpe"],
               "max_drawdown": m["max_drawdown"], "avg_weekly_turnover": m["avg_weekly_turnover"],
               "avg_exposure": m["avg_exposure"]}
        row.update(st)
        if ve is not None:
            row.update(ve[per])
        row["note"] = M.RF_NOTE
        rows.append(row)
    return rows, res


def exposure_2020(res, portfolio, option, s):
    """實驗 3 的對照表：2020-02 – 2020-04 的平均／最低曝險、期間報酬和最大回撤。"""
    a, b = STRESS["stress_2020"]
    rb = res["rebal"].loc[a:b]
    st = M.stress(res["daily"], {"stress_2020": (a, b)})
    return {"portfolio": portfolio, "option": option,
            "target": s["exposure"]["target"] if s["exposure"] else np.nan,
            "avg_exposure_2020": float(rb["exposure_target"].mean()),
            "min_exposure_2020": float(rb["exposure_target"].min()), **st}


def pick(rows, series, portfolio, base_option, exclude=()):
    """選擇規則：選項要同時滿足
    (1) 驗證期 Sharpe 比基準高至少 0.05，(2) 訓練期 Sharpe 也比基準高；
    符合的選項中取驗證期 Sharpe 最高的，沒有就保留基準。exclude：只當對照、不參加選擇的選項。"""
    df = pd.DataFrame(rows)
    d = df[(df.series == series) & (df.portfolio == portfolio) & ~df.option.isin(exclude)]
    va = d[d.period == "valid"].set_index("option")["sharpe"]
    tr = d[d.period == "train"].set_index("option")["sharpe"]
    ok = (va - va[base_option] >= SEL_MARGIN) & (tr > tr[base_option])
    chosen = va[ok].idxmax() if ok.any() else base_option
    best = va.idxmax()
    return chosen, {"series": series, "portfolio": portfolio, "base": base_option,
                    "base_valid_sharpe": va[base_option], "base_train_sharpe": tr[base_option],
                    "best_valid": best, "best_valid_sharpe": va[best], "best_train_sharpe": tr[best],
                    "qualified": ",".join(va[ok].index), "chosen": chosen,
                    "chosen_valid_sharpe": va[chosen], "chosen_train_sharpe": tr[chosen]}


def exclusion_report(ctx, results):
    """每一次成份股剔除和下市，各組在剔除前是否持有、實際賣出（或轉現金）的日期。"""
    per = dl.membership_periods()
    rows = []
    for g, res in results.items():
        W = res.get("weights")
        if W is None or not len(W):
            continue
        sig = ctx.signal[ctx.rebal.get_indexer(W.index)]
        for _, r in per[per["end"].notna() & (per["end"] >= sig[0])].iterrows():
            t = r["ds_ticker"]
            before = W.index[sig <= r["end"]]
            if not len(before) or W.loc[before[-1], t] <= 0:
                continue
            after = W.index[sig > r["end"]]
            L = pd.Timestamp(C.DELISTED[t]) if t in C.DELISTED else None
            sold = next((d for d in after if W.loc[d, t] <= 0), None)
            if L is not None and L <= (after[0] if len(after) else L):
                how, when = "下市轉現金", L
            else:
                how, when = "換倉日賣出", sold
            rows.append({"portfolio": g, "ticker": t, "last_in_index": r["end"].date(), "event": how,
                         "exit_date": when.date() if when is not None else None,
                         "first_rebal_after": after[0].date() if len(after) else None,
                         "delayed": bool(how == "換倉日賣出" and len(after) and when != after[0])})
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out("table", "exclusion_sales.csv"), index=False)
    return out_df


# ====================================================================== 主程式
def main():
    t_start = time.time()
    for d in [C.DATA_DIR, C.TABLE_DIR, C.FIG_DIR, C.LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    np.random.seed(C.SEED)

    # ---------------- 0. 資料截斷
    panel, sti = load_truncated()
    print(f"資料截斷：panel 最後日期 {panel.index[-1].date()}，STI 最後日期 {sti.index[-1].date()}"
          f"（FREEZE_DATE = {C.FREEZE_DATE}）")
    assert panel.index[-1] <= pd.Timestamp(C.FREEZE_DATE) and sti.index[-1] <= pd.Timestamp(C.FREEZE_DATE)

    ctx = Context(panel, sti)
    if ctx.screen_table is not None:
        ctx.screen_table.to_csv(out("table", "screen_static_train.csv"))
    print(f"IC 篩選（{C.SCREEN_MODE}，NW t > {C.SCREEN_T_MIN}）："
          + ("每期在訓練窗口內篩選" if ctx.screen is None else f"{len(ctx.screen)} 個因子 {ctx.screen}"))

    # ---------------- 1. 模型
    print("滾動模型：")
    ctx.fit_models()
    j0 = ctx.j0
    print(f"  第一個有 Lasso 分數的訊號日 {ctx.signal[j0].date()}（換倉 {ctx.rebal[j0].date()}）；"
          f"所有組合從這天開始回測")
    lg = ctx.model_logs["lasso260"].copy()
    names = np.array(ctx.factor_names)
    lg["screened"] = lg.pop("screen_idx").apply(lambda ix: ",".join(names[ix]))
    lg = lg.rename(columns={f"t_{k}": f"t_{n}" for k, n in enumerate(names)})
    lg.to_csv(out("table", "lasso_log.csv"), index=False, date_format="%Y-%m-%d")
    # 滾動 IC 篩選：各因子通過 t > SCREEN_T_MIN 的期數比例（訓練期 / 驗證期 / 全部）
    passed = pd.DataFrame({n: lg["screened"].str.split(",").apply(lambda v: n in v) for n in names})
    sig = pd.to_datetime(lg["signal"])
    tr, va = sig <= pd.Timestamp(C.TRAIN_END), (sig >= pd.Timestamp(C.VALID_START)) & (sig <= pd.Timestamp(C.VALID_END))
    screen_freq = pd.DataFrame({"pass_train": passed[tr.values].mean(), "pass_valid": passed[va.values].mean(),
                                "pass_all": passed.mean()})
    screen_freq.to_csv(out("table", "screen_freq.csv"))
    print("IC 篩選（%s，NW t > %.1f）通過比例：" % (C.SCREEN_MODE, C.SCREEN_T_MIN))
    print(screen_freq.round(2).sort_values("pass_all", ascending=False).to_string())
    ctx.model_coefs["lasso260"].set_axis(ctx.factor_names, axis=1).to_csv(out("table", "lasso_coef.csv"), date_format="%Y-%m-%d")
    ctx.model_coefs["enet"].set_axis(ctx.factor_names, axis=1).to_csv(out("table", "enet_coef.csv"), date_format="%Y-%m-%d")
    sc_long = pd.concat({m: pd.DataFrame(ctx.scores[m], index=ctx.signal, columns=ctx.tickers).stack()
                         for m in ["lasso260", "enet", "ew5", "fep", "lasso156", "lasso_exp"]}, axis=1)
    sc_long.index.names = ["signal", "ticker"]
    sc_long.to_parquet(out("data", "scores.parquet"))

    # Lasso 入選頻率
    coef = ctx.model_coefs["lasso260"].set_axis(ctx.factor_names, axis=1)
    freq = pd.DataFrame({
        "select_freq_train": (coef.loc[:C.TRAIN_END] > 0).mean(), "select_freq_valid": (coef.loc[C.VALID_START:C.VALID_END] > 0).mean(),
        "select_freq_all": (coef > 0).mean(), "mean_coef_all": coef.mean()})
    freq.to_csv(out("table", "lasso_selection_freq.csv"))

    # 模型的樣本外 IC
    ic_rows = []
    for m in ["lasso260", "enet", "ew5", "fep", "lasso156", "lasso_exp"]:
        sc = ctx.scores[m].copy(); sc[:j0] = np.nan
        ic = MD.score_ic(sc, ctx.hold_ret, ctx.tradable, ctx.signal)
        for per, (a, b) in PERIODS.items():
            x = ic.loc[a:b]
            ic_rows.append({"model": m, "period": per, "n_weeks": len(x), "IC_mean": x.mean(), "IC_std": x.std(),
                            "NW_t": IC.nw_tstat(x), "IC_pos_pct": (x > 0).mean()})
    model_ic = pd.DataFrame(ic_rows)
    model_ic.to_csv(out("table", "model_ic.csv"), index=False)

    # ---------------- 2. 8 組組合
    print("8 組組合回測：")
    b0_ret = sti_daily(ctx)
    groups = {
        "B0'": spec(universe="all", weighting="mv", stock_cap=None, sector_over=None, band=None),
        "B1": spec(universe="all", weighting="ew"),
        "A1": spec(universe="all", weighting="invvol"),
        "A2": with_exp(spec(universe="all", weighting="invvol")),
        "S1": spec(weighting="ew"),
        "S2": spec(weighting="invvol"),
        "S3": with_exp(spec(weighting="invvol")),
    }
    results = {}
    for g, s in groups.items():
        t0 = time.time()
        results[g] = run_strategy(ctx, s)
        print(f"  {g}（{time.time() - t0:.0f}s）")
    start_day = results["S3"]["daily"].index[0]
    results["B0"] = BT.run_index(ctx.days, b0_ret, start_day, ctx.days[-1])
    b0 = results["B0"]["daily"]["ret"]
    perf_rows = []
    for g in ["B0", "B0'", "B1", "A1", "A2", "S1", "S2", "S3"]:
        perf_rows += evaluate(ctx, results[g], g, b0)
    perf = pd.DataFrame(perf_rows)
    perf["note"] = M.RF_NOTE
    cols = ["portfolio", "period", "cost"] + [c for c in perf.columns if c not in ("portfolio", "period", "cost")]
    perf[cols].to_csv(out("table", "perf_main.csv"), index=False)

    # 股票池效果、策略效果
    eff = []
    for per in PERIODS:
        p = perf[(perf.period == per) & (perf.cost == "net")].set_index("portfolio")["ann_return"]
        eff.append({"period": per, "universe_effect_B0p_minus_B0": p["B0'"] - p["B0"],
                    "strategy_effect_S3_minus_B0p": p["S3"] - p["B0'"], "note": M.RF_NOTE})
    pd.DataFrame(eff).to_csv(out("table", "effects.csv"), index=False)

    # 每日淨值、權重、交易、曝險
    nav = pd.DataFrame({g: r["daily"]["nav"] for g, r in results.items()})
    nav.to_csv(out("table", "nav.csv"), date_format="%Y-%m-%d")
    note = f"Data {nav.index[0]:%Y-%m-%d} – {nav.index[-1]:%Y-%m-%d}; one-way cost {C.COST_BPS} bps; {M.RF_NOTE_EN}"
    PL.nav_chart(nav, out("fig", "nav_stage2.png"), note)
    PL.drawdown_chart(nav, out("fig", "drawdown_stage2.png"), note)
    PL.cost_chart(perf, out("fig", "gross_vs_net_stage2.png"), note)
    pd.concat({g: r["trades"] for g, r in results.items() if len(r["trades"])}, names=["portfolio"]).to_csv(
        out("table", "trades.csv"), date_format="%Y-%m-%d")
    pd.concat({g: r["weights"] for g, r in results.items() if len(r["weights"])}, names=["portfolio"]).to_parquet(
        out("data", "weights.parquet"))
    for g in ["A2", "S3"]:
        e = results[g]["targets"]["exposure"].copy()
        e.index = ctx.signal[j0:j0 + len(e)]
        e.to_csv(out("table", f"exposure_{g}.csv"), date_format="%Y-%m-%d")

    # 診斷表
    diag_rows, secw = [], {}
    for g in ["B0'", "B1", "A1", "A2", "S1", "S2", "S3"]:
        rb, W = results[g]["rebal"], results[g]["weights"]
        sw = pd.DataFrame({s: W.loc[:, ctx.sec == k].sum(axis=1) for k, s in enumerate(P.SECTORS)})
        secw[g] = sw
        for per, (a, b) in PERIODS.items():
            r = rb.loc[a:b]; s_ = sw.loc[a:b]
            row = {"portfolio": g, "period": per, "avg_max_weight": r["max_weight"].mean(), "max_max_weight": r["max_weight"].max(),
                   "avg_min_sigma_ann": r["min_sigma"].mean() * np.sqrt(252), "avg_hhi": r["hhi"].mean(),
                   "avg_n_hold": r["n_hold"].mean(), "n_frozen_events": int(r["n_frozen"].sum()),
                   "n_blocked_events": int(r["n_blocked"].sum()), "n_cap_relaxed": int((r["cap_status"] != "ok").sum())}
            row.update({f"sector_{s}": s_[s].mean() for s in P.SECTORS})
            diag_rows.append(row)
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(out("table", "diagnostics.csv"), index=False)
    pd.concat({g: pd.concat([results[g]["rebal"], secw[g]], axis=1) for g in secw}, names=["portfolio"]).to_csv(
        out("table", "diagnostics_by_rebal.csv"), date_format="%Y-%m-%d")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, g in zip(axes, ["B0'", "A1", "S3"]):
        secw[g].plot.area(ax=ax, linewidth=0, legend=(g == "B0'"))
        ax.set_title(f"{g} sector weights"); ax.set_ylim(0, 1.0)
    axes[0].legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(out("fig", "sector_weights.png"), dpi=120); plt.close(fig)

    exr = exclusion_report(ctx, results)
    print(f"剔除／下市事件（各組持有中的）：{len(exr)} 筆，延後賣出 {int(exr['delayed'].sum()) if len(exr) else 0} 筆")

    # Brinson（S3 相對 B0'）
    rzo = PP.rz_open(panel, ctx.days)
    reb_idx = results["S3"]["weights"].index
    nxt = list(reb_idx[1:]) + [None]
    pr = pd.DataFrame([(rzo.loc[n] / rzo.loc[d] - 1).values if n is not None else np.full(ctx.N, np.nan)
                       for d, n in zip(reb_idx, nxt)], index=reb_idx, columns=ctx.tickers)
    br = M.brinson(results["S3"]["weights"], results["B0'"]["weights"].reindex(reb_idx), pr,
                   [C.SECTOR[t] for t in ctx.tickers])
    br.to_csv(out("table", "brinson_weekly.csv"), date_format="%Y-%m-%d")
    br_sum = []
    for per, (a, b) in PERIODS.items():
        x = br.loc[a:b]
        br_sum.append({"period": per, "allocation_ann": x["allocation"].mean() * 52,
                       "selection_ann": x["selection"].mean() * 52, "active_ann": x["active"].mean() * 52,
                       "note": "S3 股票部位相對 B0'，不含現金/曝險效果；" + M.RF_NOTE})
    pd.DataFrame(br_sum).to_csv(out("table", "brinson_summary.csv"), index=False)

    # ---------------- 3. 對照模型（S3 換分數來源）
    print("對照模型：")
    model_rows = []
    for m in ["lasso260", "enet", "ew5", "fep"]:
        r = results["S3"] if m == "lasso260" else run_strategy(ctx, with_exp(spec(model=m)))
        model_rows += [dict(row, model=m) for row in evaluate(ctx, r, f"S3[{m}]", b0, gross_too=False)]
    pm = pd.DataFrame(model_rows); pm["note"] = M.RF_NOTE
    pm.to_csv(out("table", "perf_models.csv"), index=False)

    # ---------------- 4. 波動率實驗：E1（σ̂ 估計方法）、E2（低波動傾斜 γ）、V1（目標波動），一次只改一項
    print("波動率實驗：")
    vrows, choices = [], []
    E = {
        "E1": [("ewma94", {}), ("sd20", dict(sigma="sd20")), ("sd60", dict(sigma="sd60")),
               ("ewma97", dict(sigma="ewma97")), ("park20", dict(sigma="park20")), ("garch", dict(sigma="garch"))],
        "E2": [("gamma1", {}), ("gamma0", dict(gamma=0.0)), ("gamma0.5", dict(gamma=0.5)), ("gamma2", dict(gamma=2.0))],
    }
    for ser, opts in E.items():
        for pname, base_kw in [("S2", {}), ("A1", dict(universe="all"))]:
            for opt, kw in opts:
                t0 = time.time()
                s = spec(**base_kw); s.update(kw)
                rows, _ = experiment_rows(ctx, b0, ser, opt, pname, s, is_base=(opt == opts[0][0]))
                vrows += rows
                print(f"  {ser} {pname} {opt}（{time.time() - t0:.0f}s）")
        ch, info = pick(vrows, ser, "S2", opts[0][0]); choices.append(info)
    Vs = {
        "V1": [("target12", {}), ("target10", dict(target=0.10)), ("target15", dict(target=0.15))],
    }
    expo_2020 = []
    for ser, opts in Vs.items():
        for pname, base_kw in [("S3", {}), ("A2", dict(universe="all"))]:
            runs = [(opt, with_exp(spec(**base_kw), **kw)) for opt, kw in opts]
            if ser == "V1":                       # 對照：不設目標（曝險恆為 100%，= S2 / A1），不參加選擇
                runs.append(("no_target", spec(**base_kw)))
            for opt, s in runs:
                t0 = time.time()
                rows, res = experiment_rows(ctx, b0, ser, opt, pname, s, is_base=(opt == opts[0][0]), extras=True)
                vrows += rows
                if ser == "V1":
                    expo_2020.append(exposure_2020(res, pname, opt, s))
                print(f"  {ser} {pname} {opt}（{time.time() - t0:.0f}s）")
        ch, info = pick(vrows, ser, "S3", opts[0][0], exclude=("no_target",)); choices.append(info)
    pd.DataFrame(expo_2020).assign(note=M.RF_NOTE).to_csv(out("table", "vol_target_2020.csv"), index=False)

    # ---------------- 5. K、SECTOR_OVER（S3，驗證期；LASSO_WINDOW 固定 260）
    print("K、SECTOR_OVER：")
    PARAMS = {
        "K": [("12", {}), ("8", dict(K=8)), ("10", dict(K=10)), ("15", dict(K=15))],
        "SECTOR_OVER": [("0.10", {}), ("0.15", dict(sector_over=0.15))],
    }
    for ser, opts in PARAMS.items():
        for opt, kw in opts:
            t0 = time.time()
            s = with_exp(spec(**kw))
            rows, _ = experiment_rows(ctx, b0, ser, opt, "S3", s, is_base=(opt == opts[0][0]))
            vrows += rows
            print(f"  {ser}={opt}（{time.time() - t0:.0f}s）")
        ch, info = pick(vrows, ser, "S3", opts[0][0]); choices.append(info)
    vx = pd.DataFrame(vrows)
    vx.to_csv(out("table", "vol_experiments.csv"), index=False)
    ch_df = pd.DataFrame(choices)
    ch_df.to_csv(out("table", "param_selection.csv"), index=False)

    # γ 曲線（E2）
    PL.gamma_chart(vx, out("fig", "gamma_curve.png"), f"Validation {C.VALID_START} – {C.VALID_END}; {M.RF_NOTE_EN}")

    # ---------------- 6. 合併選定的設定 → frozen_params.json
    chosen = ch_df.set_index("series")["chosen"].to_dict()
    final = spec()
    e_final = dict(EXP_BASE)
    final_notes = ["波動率實驗 E1、E2、V1，加上 K、SECTOR_OVER；LASSO_WINDOW 固定 260"]
    opt_kw = {ser: dict(opts) for ser, opts in {**E, **Vs, **PARAMS}.items()}
    for ser in ["E1", "E2", "K", "SECTOR_OVER"]:
        final.update(opt_kw[ser][chosen[ser]])
    for ser in ["V1"]:
        e_final.update(opt_kw[ser][chosen[ser]])
    final_s3 = dict(final); final_s3["exposure"] = e_final
    res_final = run_strategy(ctx, final_s3)
    fin_rows = evaluate(ctx, res_final, "S3_final", b0)
    pd.DataFrame(fin_rows).assign(note=M.RF_NOTE).to_csv(out("table", "perf_final_S3.csv"), index=False)
    frozen = {
        "freeze_date": C.FREEZE_DATE, "selection_rule": f"每個系列一次只改一項；選項必須同時滿足：驗證期（{C.VALID_START}–{C.VALID_END}）Sharpe "
        f"比基準高至少 {SEL_MARGIN}，而且訓練期 Sharpe 也比基準高；符合者取驗證期 Sharpe 最高，否則保留基準",
        "chosen": chosen, "notes": final_notes,
        "params": {"K": final["K"], "RANK_BUFFER": final["buffer"], "LASSO_MODEL": final["model"],
                   "LASSO_WINDOW": {"lasso260": 260, "lasso156": 156, "lasso_exp": "expanding"}[final["model"]],
                   "LASSO_MIN_WINDOW": C.LASSO_MIN_WINDOW, "LAMBDA_VALID_WEEKS": C.LAMBDA_VALID_WEEKS,
                   "WEIGHTING": final["weighting"], "GAMMA": final["gamma"], "SIGMA_METHOD": final["sigma"],
                   "SIGMA_FLOOR_PCT": C.SIGMA_FLOOR_PCT, "MAX_WEIGHT": final["stock_cap"],
                   "SECTOR_OVER": final["sector_over"], "NO_TRADE_BAND": final["band"],
                   "EXPOSURE": e_final, "MAX_EXPOSURE": C.MAX_EXPOSURE, "COST_BPS": final["cost_bps"],
                   "SCREEN_MODE": C.SCREEN_MODE, "SCREEN_T_MIN": C.SCREEN_T_MIN, "SCREEN_FACTORS": ctx.screen},
    }
    with open(out("log", "../frozen_params.json"), "w") as f:
        json.dump(frozen, f, ensure_ascii=False, indent=2, default=str)

    # 預處理狀態（截斷後）
    ctx.prep_status.groupby(["factor", "status"]).size().unstack(fill_value=0).to_csv(
        out("table", "stage2_preprocess_status.csv"))
    with open(out("log", "stage2_manifest.json"), "w") as f:
        json.dump(sorted(set(MANIFEST)), f, ensure_ascii=False, indent=1)

    # ---------------- 7. 印出摘要
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    show = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "n_drawdowns_5pct", "info_ratio", "avg_weekly_turnover",
            "avg_exposure", "n_trades", "win_rate", "avg_hold_days", "avg_trade_pnl", "alpha", "alpha_t", "beta"]
    for per in ["train", "valid", "full"]:
        print(f"\n=== 績效（{per}，扣成本後；r_f = 0）===")
        print(perf[(perf.period == per) & (perf.cost == "net")].set_index("portfolio")[show].round(3))
    print("\n=== 模型樣本外 IC ===\n", model_ic.round(4))
    print("\n=== Lasso 入選頻率 ===\n", freq.round(3))
    print("\n=== 診斷表（valid）===\n", diag[diag.period == "valid"].set_index("portfolio").round(3).T)
    print("\n=== 參數選擇 ===\n", ch_df.round(3))
    print("\n=== 最終 S3 ===\n", pd.DataFrame(fin_rows).set_index(["period", "cost"])[show].round(3))
    print(f"\n總時間 {time.time() - t_start:.0f}s")
    return ctx, results


if __name__ == "__main__":
    main()
