"""月頻對照實驗：只改換倉頻率，其他凍結參數全部沿用，跟週頻主設定比較。

這是對照實驗，不參與參數選擇：讀 outputs/frozen_params.json，不寫回。資料跟階段 2 一樣截斷在
FREEZE_DATE（2026-09-30），只算訓練期、驗證期、全期間，不跑測試期。
週頻用同一份凍結參數重跑（結果確定），並檢查跟 perf_frozen_history.csv 一致。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd

import config as C
import data_loader as dl
import metrics as M
import model as MD
import plots as PL
import preprocess as PP
import run_stage2 as R2
import run_stage3 as R3
import volatility as V

GROUPS = ["B1", "A1", "A2", "S1", "S2", "S3"]
EXPO_GROUPS = ("A2", "S3")
# 月頻專屬設定（6.7 的表）；其他參數一律來自 frozen_params.json
MONTHLY = dict(window=60, min_window=36, val_periods=24, garch_horizon=21)
OUT = []


def out(kind, name):
    d = {"table": C.TABLE_DIR, "fig": C.FIG_DIR, "log": C.LOG_DIR}[kind]
    OUT.append(os.path.relpath(os.path.join(d, name), C.ROOT))
    return os.path.join(d, name)


# ====================================================================== 月頻日曆與資料
def monthly_index(weekly_sched):
    """週頻日曆中，每個曆月日期最早的那一個換倉日的列位置。"""
    reb = pd.DatetimeIndex(weekly_sched["rebal"])
    s = pd.Series(np.arange(len(reb)), index=reb)
    return s.groupby(reb.to_period("M")).min().values


class MonthlyView:
    """把週頻 Context 改成月頻換倉的介面，給 R2.build_targets / R2.run_strategy 用。
    z 值是每個訊號日各自的截面處理，跟頻率無關，直接取週頻的那幾列；y 改成「換倉日開盤 → 下個月換倉日開盤」。"""

    def __init__(self, ctx, model_name):
        self.ctx = ctx
        self.widx = monthly_index(ctx.sched)
        w = ctx.sched.iloc[self.widx].reset_index(drop=True)
        self.sched = pd.DataFrame({"signal": w["signal"], "rebal": w["rebal"]})
        self.sched["next_rebal"] = self.sched["rebal"].shift(-1)
        self.signal = pd.DatetimeIndex(self.sched["signal"])
        self.rebal = pd.DatetimeIndex(self.sched["rebal"])
        self.y_end = pd.DatetimeIndex(self.sched["next_rebal"]).values
        for a in ["days", "mkt", "N", "sec", "r", "sti_ret", "ph", "pl", "traded", "tickers", "factor_names", "screen"]:
            setattr(self, a, getattr(ctx, a))
        self.sig_pos = ctx.sig_pos[self.widx]
        self.tradable = ctx.tradable[self.widx]
        self.mv = ctx.mv[self.widx]
        self.X = ctx.X[self.widx]
        rzo = PP.rz_open(ctx.panel, ctx.days)
        self.hold_ret = PP.holding_returns(rzo, self.sched, 1, rz=dl.field(ctx.panel.loc[ctx.days], "RZ")).values
        self.y = MD.demeaned_target(self.hold_ret, self.tradable)
        keep = [self.factor_names.index(f) for f in self.screen]
        sc, coef, log = MD.rolling_model(self.X, self.y, self.tradable, self.signal.values, self.y_end, kind="lasso",
                                         window=MONTHLY["window"], min_window=MONTHLY["min_window"],
                                         val_weeks=MONTHLY["val_periods"], keep=keep)
        self.scores = {model_name: sc}
        self.model_coefs = {model_name: coef}
        self.model_logs = {model_name: log}
        self.j0 = int(np.where(np.isfinite(sc).any(axis=1))[0][0])
        self._sigma = {}

    def sigma(self, method):
        if method not in self._sigma:
            raw = V.stock_sigma(self.r, self.sig_pos, method, ph=self.ph, pl=self.pl, traded=self.traded)
            elig = self.tradable & np.isfinite(raw)
            self._sigma[method] = V.floor_sigma(raw, elig)
        return self._sigma[method]


def monthly_specs(params):
    """月頻設定 = 凍結設定，只有 GARCH 預測期間改成 21 天（Lasso 窗口在 MonthlyView 設定）。"""
    specs = R3.frozen_specs(params)
    out_ = {}
    for g in GROUPS:
        s = dict(specs[g])
        if s["exposure"] is not None:
            s["exposure"] = dict(s["exposure"], horizon=MONTHLY["garch_horizon"])
        out_[g] = s
    return out_


# ====================================================================== 指標
def rows_for(res, freq, b0, sti):
    rows = []
    for g in GROUPS:
        r = res[g]
        for per, (a, b) in R2.PERIODS.items():
            rb = r["rebal"].loc[a:b]
            d = r["daily"].loc[a:b]
            years = len(d) / 252
            for gross in (False, True):
                m = M.perf(r["daily"], b0, sti, r["trades"], r["rebal"], a, b, gross=gross)
                st = M.stress(r["daily"].assign(ret=r["daily"]["ret_gross"]) if gross else r["daily"], R2.STRESS)
                rows.append({
                    "portfolio": g, "freq": freq, "period": per, "cost": "gross" if gross else "net",
                    "start": m["start"], "end": m["end"],
                    "ann_return": m["ann_return"], "ann_vol": m["ann_vol"], "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"],
                    "avg_turnover_per_rebal": float(rb["one_way_turnover"].mean()),
                    "ann_turnover": float(rb["one_way_turnover"].sum() / years),
                    "ann_cost": float(rb["cost"].sum() / years),
                    "avg_exposure": float(rb["stock_weight"].mean()) if g in EXPO_GROUPS else np.nan,
                    **st, "n_rebal": len(rb), "note": M.RF_NOTE})
    return rows


def lasso_freq(coef, label):
    coef = coef.dropna(how="all")                      # 訓練資料不足、沒有估計的期數不計
    sig = pd.DatetimeIndex(coef.index)
    nz = coef > 0
    tr = sig <= pd.Timestamp(C.TRAIN_END)
    va = (sig >= pd.Timestamp(C.VALID_START)) & (sig <= pd.Timestamp(C.VALID_END))
    return pd.DataFrame({f"{label}_train": nz[tr].mean(), f"{label}_valid": nz[va].mean(),
                         f"{label}_mean_coef_train": coef[tr].mean(), f"{label}_mean_coef_valid": coef[va].mean()})


# ====================================================================== 主程式
def build(params):
    panel, sti = R2.load_truncated()
    ctx = R2.Context(panel, sti, verbose=False, screen=params["SCREEN_FACTORS"])
    ctx.fit_models(which=(params["LASSO_MODEL"],))
    mv = MonthlyView(ctx, params["LASSO_MODEL"])
    return ctx, mv


def main():
    t0 = time.time()
    np.random.seed(C.SEED)
    with open(os.path.join(C.OUT_DIR, "frozen_params.json")) as f:
        frozen = json.load(f)
    params = frozen["params"]
    ctx, mv = build(params)
    print(f"資料截斷：最後交易日 {ctx.days[-1].date()}（FREEZE_DATE = {C.FREEZE_DATE}）")
    print(f"週頻換倉 {len(ctx.rebal)} 次；月頻換倉 {len(mv.rebal)} 次，第一個月頻分數 {mv.signal[mv.j0].date()}"
          f"（週頻 {ctx.signal[ctx.j0].date()}）")

    sti = pd.Series(ctx.sti_ret, index=ctx.days)
    wk = R3.run_groups(ctx, {g: s for g, s in R3.frozen_specs(params).items() if g in GROUPS}, C.FREEZE_DATE)
    mo = R3.run_groups(mv, monthly_specs(params), C.FREEZE_DATE)
    # 相對 B0 的指標不輸出，B0 只是 perf 的必要輸入：兩種頻率都用週頻那一份
    b0 = wk["B0"]["daily"]["ret"]
    df = pd.DataFrame(rows_for(wk, "weekly", b0, sti) + rows_for(mo, "monthly", b0, sti))
    df.to_csv(out("table", "freq_compare.csv"), index=False)

    names = ctx.factor_names                            # 係數表的欄位是因子序號
    las = lasso_freq(ctx.model_coefs[params["LASSO_MODEL"]].set_axis(names, axis=1), "weekly").join(
        lasso_freq(mv.model_coefs[params["LASSO_MODEL"]].set_axis(names, axis=1), "monthly"))
    las = las.loc[params["SCREEN_FACTORS"]]
    las["note"] = "入選 = 係數 > 0 的期數比例；週頻 260 期窗口，月頻 60 期窗口"
    las.to_csv(out("table", "freq_compare_lasso.csv"))

    PL.freq_chart(wk["S3"]["daily"]["nav"], mo["S3"]["daily"]["nav"], wk["B0"]["daily"]["nav"],
                  out("fig", "freq_compare_S3.png"),
                  f"Frozen parameters; monthly = first weekly rebalance date of each month; data to {C.FREEZE_DATE}; 15 bps; {M.RF_NOTE_EN}")
    with open(out("log", "freq_compare_manifest.json"), "w") as f:
        json.dump(sorted(set(OUT)), f, ensure_ascii=False, indent=1)

    pd.set_option("display.width", 220)
    show = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_turnover_per_rebal", "ann_turnover", "ann_cost",
            "avg_exposure", "n_rebal"]
    for per in ["train", "valid", "full"]:
        x = df[df.period == per].set_index(["portfolio", "cost", "freq"])[show].round(3).sort_index()
        print(f"\n=== {per} ===\n{x}")
    print("\n=== Lasso 入選頻率 ===\n", las.round(3))
    print(f"\n總時間 {time.time() - t0:.0f}s")
    return ctx, mv, wk, mo, df


if __name__ == "__main__":
    main()
