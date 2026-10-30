"""階段 3：用凍結參數跑測試期（2026-10-01 起到資料最後一天），敏感度測試，報告圖表與繳交用 CSV。

測試期：10/1（週四）開盤用 9/30 收盤的訊號從現金初始建倉，之後每週二（遇假日順延）開盤換倉，
持有到資料最後一天收盤（資料重抓到 10/29 後再跑一次；測試期只到 config.TEST_END = 10/29）。參數全部來自 outputs/frozen_params.json，
不得根據測試期結果回頭調整。
"""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd

import backtest as BT
import config as C
import data_loader as dl
import metrics as M
import model as MD
import plots as PL
import portfolio as P
import preprocess as PP
import run_stage2 as R2
import volatility as V

GROUPS = ["B0", "B0'", "B1", "A1", "A2", "S1", "S2", "S3"]
TEST = (C.TEST_START, C.TEST_END)
OUT = []


def out(kind, name):
    d = {"table": C.TABLE_DIR, "fig": C.FIG_DIR, "data": C.DATA_DIR, "log": C.LOG_DIR}[kind]
    OUT.append(os.path.relpath(os.path.join(d, name), C.ROOT))
    return os.path.join(d, name)


# ====================================================================== 凍結參數
def load_frozen():
    with open(os.path.join(C.OUT_DIR, "frozen_params.json")) as f:
        return json.load(f)


def frozen_specs(params, **override):
    """8 組的設定：所有非基準組都套用凍結的 K、σ̂ 方法、加權方式、上限、不交易區間、成本。"""
    p = dict(params)
    base = R2.spec(K=p["K"], buffer=p["RANK_BUFFER"], model=p["LASSO_MODEL"], weighting=p["WEIGHTING"],
                   gamma=p["GAMMA"], sigma=p["SIGMA_METHOD"], stock_cap=p["MAX_WEIGHT"],
                   sector_over=p["SECTOR_OVER"], band=p["NO_TRADE_BAND"], cost_bps=p["COST_BPS"])
    base.update({k: v for k, v in override.items() if k != "exposure"})
    expo = dict(p["EXPOSURE"]); expo.update(override.get("exposure", {}))
    w = lambda **kw: dict(base, **kw)
    return {
        "B0'": w(universe="all", weighting="mv", stock_cap=None, sector_over=None, band=None),
        "B1": w(universe="all", weighting="ew"),
        "A1": w(universe="all"),
        "A2": w(universe="all", exposure=expo),
        "S1": w(weighting="ew"),
        "S2": w(),
        "S3": w(exposure=expo),
    }


# ====================================================================== 測試期的訊號日與換倉日
def test_schedule(days, weekly, start=C.TEST_START, end=C.TEST_END):
    """第一個換倉日 = start 當天或之後第一個交易日（初始建倉）；之後是每週的正常換倉日（> 初始建倉日、≤ end、≤ 資料最後一天）。
    訊號日 = 換倉日前一個交易日。"""
    first = days[days >= pd.Timestamp(start)][0]
    reb = [first] + [r for r in pd.DatetimeIndex(weekly["rebal"]) if first < r <= pd.Timestamp(end) and r <= days[-1]]
    reb = pd.DatetimeIndex(reb)
    sig = days[days.get_indexer(reb) - 1]
    return pd.DataFrame({"signal": sig, "rebal": reb})


class TestView:
    """把完整資料的 Context 改成「只在測試期的訊號日出訊號」的介面，給 R2.build_targets / run_strategy 用。
    模型仍然用週頻的歷史訓練資料（ctx.X、ctx.y），只用持有期結束 ≤ 訊號日的期數。"""

    def __init__(self, ctx, sched, model_names):
        self.ctx = ctx
        self.signal = pd.DatetimeIndex(sched["signal"])
        self.rebal = pd.DatetimeIndex(sched["rebal"])
        self.days, self.mkt, self.N, self.sec = ctx.days, ctx.mkt, ctx.N, ctx.sec
        self.r, self.sti_ret, self.ph, self.pl, self.traded = ctx.r, ctx.sti_ret, ctx.ph, ctx.pl, ctx.traded
        self.sig_pos = ctx.days.get_indexer(self.signal)
        self.tradable = ctx.trad_daily.reindex(self.signal).fillna(False).astype(bool).values
        mv_daily = dl.field(ctx.panel.loc[ctx.days], "MV")
        self.mv = mv_daily.reindex(self.signal).values
        z, _ = PP.preprocess_all(ctx.fac, ctx.trad_daily, mv_daily, self.signal)
        self.X = np.stack([z[k].fillna(0.0).values for k in ctx.factor_names], axis=2)
        self.j0 = 0
        self.scores, self.coefs, self.logs = {}, {}, {}
        keep = None if ctx.screen is None else [ctx.factor_names.index(f) for f in ctx.screen]
        ic = MD.weekly_factor_ic(ctx.X, ctx.y, ctx.tradable) if keep is None else None   # 滾動篩選才需要
        for m in model_names:
            kind = "enet" if m == "enet" else "lasso"
            window = {"lasso260": C.LASSO_WINDOW, "enet": C.LASSO_WINDOW, "lasso156": 156,
                      "lasso_exp": "expanding"}[m]
            sc, cf, lg = [], {}, []
            for k, t in enumerate(self.signal):
                beta, log = MD.fit_at(ctx.X, ctx.y, ctx.tradable, ctx.y_end, t, kind=kind, window=window, ic=ic, keep=keep)
                sc.append(MD.score_from_beta(beta, self.X[k], self.tradable[k], log["screen_idx"]))
                cf[t] = beta; lg.append(log)
            self.scores[m] = np.array(sc)
            self.coefs[m] = pd.DataFrame(cf).T.set_axis(ctx.factor_names, axis=1)
            self.logs[m] = pd.DataFrame(lg)
        self._sigma = {}

    def sigma(self, method):
        if method not in self._sigma:
            raw = V.stock_sigma(self.r, self.sig_pos, method, ph=self.ph, pl=self.pl, traded=self.traded)
            elig = self.tradable & np.isfinite(raw)
            self._sigma[method] = V.floor_sigma(raw, elig)
        return self._sigma[method]


# ====================================================================== 執行
def run_groups(view, specs, end_date, close_exec=False):
    v = view
    if close_exec:
        v = copy.copy(view); v.mkt = view.mkt.close_execution()
    res = {}
    for g, s in specs.items():
        tg = R2.build_targets(v, s)
        n = len(tg["rel"])
        expo = tg["exposure"]["exposure"].values if tg["exposure"] is not None else None
        r = BT.run(v.mkt, v.rebal[v.j0:v.j0 + n], tg["rel"], exposure=expo, cost_bps=s["cost_bps"], band=s["band"],
                   stock_cap=s["stock_cap"], sector=v.sec,
                   sector_caps_rel=tg["caps"] if s["sector_over"] is not None else None, end_date=end_date,
                   sigma=tg["sigma"], priority=tg["priority"])
        r["targets"] = tg
        res[g] = r
    start = res[next(iter(res))]["daily"].index[0]
    res["B0"] = BT.run_index(v.days, pd.Series(v.sti_ret, index=v.days), start, end_date)
    return res


def perf_table(res, periods, sti):
    rows = []
    b0 = res["B0"]["daily"]["ret"]
    for g in GROUPS:
        if g not in res:
            continue
        for per, (a, b) in periods.items():
            for gross in [False, True]:
                m = M.perf(res[g]["daily"], b0, sti, res[g]["trades"], res[g]["rebal"], a, b, gross=gross)
                d = res[g]["daily"].loc[a:b]
                m["cum_return"] = float((1 + d["ret_gross" if gross else "ret"]).prod() - 1)
                if g == "B0":
                    m.update(alpha=np.nan, alpha_t=np.nan, beta=np.nan, beta_t=np.nan, info_ratio=np.nan,
                             tracking_error=np.nan, excess_vs_B0=np.nan)
                m.update(portfolio=g, period=per, cost="gross" if gross else "net", note=M.RF_NOTE)
                rows.append(m)
    df = pd.DataFrame(rows)
    return df[["portfolio", "period", "cost"] + [c for c in df.columns if c not in ("portfolio", "period", "cost")]]


def build_all(panel, sti, frozen, membership=C.MEMBERSHIP_MASK, exclude=()):
    """完整資料的 Context（歷史）＋ 測試期 TestView。membership 預設 = config（每期成份股）。"""
    ctx = R2.Context(panel, sti, verbose=False, membership=membership, exclude=exclude,
                     screen=frozen["params"].get("SCREEN_FACTORS"))                # 用凍結的因子清單，不重新篩選
    ctx.fit_models(which=(frozen["params"]["LASSO_MODEL"],))
    view = TestView(ctx, test_schedule(ctx.days, ctx.sched), [frozen["params"]["LASSO_MODEL"]])
    return ctx, view


def variant_rows(ctx, view, specs, label, groups, close_exec=False):
    """敏感度：歷史（2014 – 9/30）和測試期各跑一次，回傳指標列。"""
    rows = []
    sti = pd.Series(ctx.sti_ret, index=ctx.days)
    sub = {g: specs[g] for g in groups}
    hv = ctx
    if close_exec:
        hv = copy.copy(ctx); hv.mkt = ctx.mkt.close_execution()
    h = run_groups(hv, sub, C.FREEZE_DATE)
    t = run_groups(view, sub, view.days[-1], close_exec=close_exec)
    for g in groups:
        for per, (a, b) in R2.PERIODS.items():
            m = M.perf(h[g]["daily"], h["B0"]["daily"]["ret"], sti, h[g]["trades"], h[g]["rebal"], a, b)
            rows.append(dict(variant=label, portfolio=g, period=per, **_pick(m)))
        m = M.perf(t[g]["daily"], t["B0"]["daily"]["ret"], sti, t[g]["trades"], t[g]["rebal"], *TEST)
        m["cum_return"] = float((1 + t[g]["daily"]["ret"].loc[TEST[0]:TEST[1]]).prod() - 1)
        rows.append(dict(variant=label, portfolio=g, period="test", **_pick(m)))
    return rows


def _pick(m):
    keys = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_weekly_turnover", "avg_exposure", "excess_vs_B0",
            "info_ratio", "cum_return"]
    return {k: m.get(k, np.nan) for k in keys}


def main():
    t_start = time.time()
    for d in [C.TABLE_DIR, C.FIG_DIR, C.LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    np.random.seed(C.SEED)
    frozen = load_frozen()
    params = frozen["params"]
    print("凍結參數（outputs/frozen_params.json，凍結日 %s）：" % frozen["freeze_date"])
    print("  " + json.dumps(params, ensure_ascii=False))

    panel, sti = dl.load_panel(rebuild=False)
    panel, sti = panel.loc[:C.TEST_END], sti.loc[:C.TEST_END]          # 測試期只到 config.TEST_END（10/29）
    print(f"資料最後日期：panel {panel.index[-1].date()}，STI {sti.index[-1].date()}（TEST_END = {C.TEST_END}）")
    ctx, view = build_all(panel, sti, frozen)
    print(f"測試期換倉日：{[d.strftime('%m-%d') for d in view.rebal]}（訊號日 {[d.strftime('%m-%d') for d in view.signal]}）；"
          f"持有到 {ctx.days[-1].date()} 收盤")
    sti_s = pd.Series(ctx.sti_ret, index=ctx.days)
    specs = frozen_specs(params)

    # ---------------- 1. 凍結設定的歷史回測（到 9/30）與測試期
    hist = run_groups(ctx, specs, C.FREEZE_DATE)
    test = run_groups(view, specs, ctx.days[-1])
    ph = perf_table(hist, R2.PERIODS, sti_s)
    ph.to_csv(out("table", "perf_frozen_history.csv"), index=False)
    pt = perf_table(test, {"test": TEST}, sti_s)
    pt.to_csv(out("table", "perf_test.csv"), index=False)

    nav_h = pd.DataFrame({g: hist[g]["daily"]["nav"] for g in GROUPS})
    nav_t = pd.DataFrame({g: test[g]["daily"]["nav"] for g in GROUPS})
    nav_h.to_csv(out("table", "nav_frozen_history.csv"), date_format="%Y-%m-%d")
    nav_t.to_csv(out("table", "nav_test.csv"), date_format="%Y-%m-%d")
    pd.concat({g: hist[g]["weights"] for g in GROUPS if g != "B0"}, names=["portfolio"]).to_parquet(
        out("data", "weights_frozen_history.parquet"))
    pd.concat({g: test[g]["weights"] for g in GROUPS if g != "B0"}, names=["portfolio"]).to_csv(
        out("table", "weights_test.csv"), date_format="%Y-%m-%d")
    pd.concat({g: test[g]["trades"] for g in GROUPS if g != "B0" and len(test[g]["trades"])}, names=["portfolio"]).to_csv(
        out("table", "trades_test.csv"), date_format="%Y-%m-%d")
    pd.concat({g: test[g]["rebal"] for g in GROUPS if g != "B0"}, names=["portfolio"]).to_csv(
        out("table", "rebal_test.csv"), date_format="%Y-%m-%d")

    # 測試期每個訊號日的分數與選股
    m = params["LASSO_MODEL"]
    sc = pd.DataFrame(view.scores[m], index=view.signal, columns=ctx.tickers)
    sel = (test["S3"]["weights"] > 0).astype(int); sel.index = view.signal[:len(sel)]
    pd.concat({"score": sc, "held_S3": sel}, axis=1).to_csv(out("table", "test_scores.csv"), date_format="%Y-%m-%d")
    view.coefs[m].to_csv(out("table", "lasso_coef_test.csv"), date_format="%Y-%m-%d")
    view.logs[m].to_csv(out("table", "lasso_log_test.csv"), index=False, date_format="%Y-%m-%d")

    # 曝險（歷史 + 測試期）
    expo = {}
    for g in ["A2", "S3"]:
        eh = hist[g]["rebal"]["stock_weight"]; et = test[g]["rebal"]["stock_weight"]
        expo[g] = pd.concat([eh, et])
    expo = pd.DataFrame(expo)
    expo.to_csv(out("table", "exposure_frozen.csv"), date_format="%Y-%m-%d")

    # ---------------- 2. 圖
    note_h = f"Frozen settings; data {nav_h.index[0]:%Y-%m-%d} – {nav_h.index[-1]:%Y-%m-%d}; one-way cost {params['COST_BPS']} bps; {M.RF_NOTE_EN}"
    note_t = (f"Test period {nav_t.index[0]:%Y-%m-%d} – {nav_t.index[-1]:%Y-%m-%d} (initial build at the 10/1 open); frozen settings; "
              f"one-way cost {params['COST_BPS']} bps; {M.RF_NOTE_EN}")
    PL.nav_chart(nav_h, out("fig", "nav_frozen.png"), note_h, title="Cumulative NAV (frozen settings, net of costs)",
                 k_label=str(params["K"]))
    nav_t0 = pd.concat([pd.DataFrame({g: [1.0] for g in GROUPS}, index=[ctx.days[ctx.days < nav_t.index[0]][-1]]), nav_t])
    PL.nav_chart(nav_t0, out("fig", "nav_test.png"), note_t, title="Test-period cumulative NAV (net of costs)", shade_valid=False,
                 log=False, k_label=str(params["K"]))
    PL.drawdown_chart(nav_h, out("fig", "drawdown_frozen.png"), note_h)
    PL.cost_chart(ph, out("fig", "gross_vs_net_frozen.png"), note_h)
    PL.exposure_chart(expo, out("fig", "exposure.png"), f"Frozen settings (GARCH(1,1)-t, target {params['EXPOSURE']['target']:.0%}); {M.RF_NOTE_EN}",
                      test_start=C.TEST_START)
    coef_all = pd.concat([ctx.model_coefs[m].set_axis(ctx.factor_names, axis=1).loc[:C.FREEZE_DATE], view.coefs[m]])
    coef_all = coef_all[~coef_all.index.duplicated(keep="last")].sort_index()
    coef_all.to_csv(out("table", "lasso_coef_all.csv"), date_format="%Y-%m-%d")
    PL.lasso_heatmap(coef_all, out("fig", "lasso_heatmap.png"), f"{m}; last signal date {coef_all.index[-1]:%Y-%m-%d}")
    W = pd.concat([hist["S3"]["weights"], test["S3"]["weights"]])
    secw = pd.DataFrame({s: W.loc[:, ctx.sec == k].sum(axis=1) for k, s in enumerate(P.SECTORS)})
    PL.sector_chart(secw, out("fig", "sector_weights_S3.png"),
                    f"Frozen settings; rebalance dates {secw.index[0]:%Y-%m-%d} – {secw.index[-1]:%Y-%m-%d} (incl. test period); {M.RF_NOTE_EN}",
                    "S3 sector weights (within the equity position)")
    decay = pd.read_csv(os.path.join(C.TABLE_DIR, "ic_decay_train.csv"), index_col=[0, 1])
    PL.ic_decay_heatmap(decay, out("fig", "ic_decay.png"), "Training period 2011–2022; stage 1 results")

    # ---------------- 3. 繳交用 CSV：測試期成份股（10/1–10/29 任一訊號日在股票池內）每日收盤價
    # （SGD；美元報價股已用 MAS 當天匯率換算）
    names = dl.ticker_names()
    pp = panel.loc[dl.trading_days(panel)]
    test_members = dl.membership_mask(view.signal, C.MEMBERSHIP_MASK or "in_index").any()
    test_members = list(test_members.index[test_members.values])
    print(f"測試期成份股 {len(test_members)} 檔")
    up, px = dl.field(pp, "UP")[test_members], dl.field(pp, "P")[test_members]
    assets = pd.concat({"UP": up.stack(future_stack=True), "P": px.stack(future_stack=True)}, axis=1).dropna(how="all")
    assets.index.names = ["date", "ticker"]
    assets = assets.reset_index()
    assets.insert(2, "name", assets["ticker"].map(names))
    assets["currency"] = "SGD"
    assets.to_csv(out("table", "assets_daily_close.csv"), index=False, date_format="%Y-%m-%d")

    # ---------------- 4. 敏感度測試
    print("敏感度測試：")
    rows = []
    t0 = time.time()
    rows += variant_rows(ctx, view, specs, "frozen", GROUPS[1:])
    for bps in [10, 20, 30]:
        rows += variant_rows(ctx, view, frozen_specs(params, cost_bps=bps), f"cost_{bps}bps", ["S1", "S2", "S3"])
    for k in [k for k in (8, 10, 12, 15) if k != params["K"]]:   # K 格點中凍結值以外的
        rows += variant_rows(ctx, view, frozen_specs(params, K=k), f"K_{k}", ["S1", "S2", "S3"])
    for tv in [tv for tv in (0.10, 0.12, 0.15) if not np.isclose(tv, params["EXPOSURE"]["target"])]:   # 格點中凍結值以外的
        rows += variant_rows(ctx, view, frozen_specs(params, exposure={"target": tv}), f"vol_target_{tv:.2f}", ["A2", "S3"])
    rows += variant_rows(ctx, view, specs, "tuesday_close", GROUPS[1:], close_exec=True)
    print(f"  成本 / K / 目標波動 / 收盤換倉（{time.time() - t0:.0f}s）")
    # 剔除第二上市股；固定 38 檔（今天仍上市、曾為成份股的公司，用同一組凍結參數，衡量存活者偏誤）
    fixed38_ex = tuple(t for t in C.TICKERS if t not in C.FIXED38)
    for label, kw in [("ex_THBE_YSHL", dict(exclude=("THBE", "YSHL"))),
                      ("ex_IHHH_EMPE", dict(exclude=tuple(C.SECONDARY_LISTED))),
                      ("fixed38", dict(membership=None, exclude=fixed38_ex))]:
        t0 = time.time()
        c2, v2 = build_all(panel, sti, frozen, **kw)
        rows += variant_rows(c2, v2, specs, label, GROUPS[1:])
        print(f"  {label}（{time.time() - t0:.0f}s）")
    sens = pd.DataFrame(rows); sens["note"] = M.RF_NOTE
    sens.to_csv(out("table", "sensitivity.csv"), index=False)

    with open(out("log", "stage3_manifest.json"), "w") as f:
        json.dump(sorted(set(OUT)), f, ensure_ascii=False, indent=1)

    # ---------------- 5. 摘要
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    show = ["cum_return", "ann_vol", "max_drawdown", "excess_vs_B0", "avg_weekly_turnover", "avg_exposure", "n_trades"]
    print("\n=== 測試期（扣成本後；r_f = 0，現金不計息）===")
    print(pt[pt.cost == "net"].set_index("portfolio")[show].round(4))
    print("\n=== 測試期（扣成本前）===")
    print(pt[pt.cost == "gross"].set_index("portfolio")[["cum_return"]].round(4).T)
    show_h = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_weekly_turnover", "avg_exposure"]
    for per in ["train", "valid", "full"]:
        print(f"\n=== 凍結設定的歷史回測（{per}，扣成本後）===")
        print(ph[(ph.period == per) & (ph.cost == "net")].set_index("portfolio")[show_h].round(3))
    print("\n=== 測試期 S3 持股（最後一次換倉）===")
    lw = test["S3"]["weights"].iloc[-1]; lw = lw[lw > 0].sort_values(ascending=False)
    print(pd.DataFrame({"weight": lw.round(4), "name": names.reindex(lw.index), "sector": [C.SECTOR[t] for t in lw.index]}))
    print("\n=== Lasso 測試期係數 ===\n", view.coefs[m].round(5).T[view.coefs[m].T.abs().sum(axis=1) > 0])
    piv = sens.pivot_table(index=["variant", "portfolio"], columns="period", values="sharpe").round(3)
    piv["test_cum_return"] = sens[sens.period == "test"].set_index(["variant", "portfolio"])["cum_return"].round(4)
    print("\n=== 敏感度（Sharpe；測試期為累積報酬）===\n", piv)
    print(f"\n總時間 {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
