"""階段 1：讀取資料 → 股票池與遮罩 → 19 個因子 → 預處理 → 週頻 IC 檢定（只用訓練期）。

用法：
    python run_stage1.py                 # 全部 20 個因子重算
    python run_stage1.py --only new      # 只算 config.NEW_FACTORS，結果併進既有的表（其他因子沿用舊結果）
    python run_stage1.py --only a,b      # 只算指定的因子
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd

import config as C
import data_loader as dl
import factors as F
import ic_test as IC
import preprocess as PP


def _path(kind, name):
    return os.path.join(C.DATA_DIR if kind == "data" else C.TABLE_DIR, name)


def _merge_rows(new, name, index_cols=1):
    """把新因子的列併進既有的 csv（同名的列以新結果取代），依 config.FACTORS 排序。"""
    path = _path("table", name)
    if os.path.exists(path):
        old = pd.read_csv(path, index_col=list(range(index_cols)))
        old = old[~old.index.isin(new.index)] if index_cols == 1 else \
            old[~old.index.get_level_values(0).isin(new.index.get_level_values(0))]
        new = pd.concat([old, new])
    key = new.index if index_cols == 1 else new.index.get_level_values(0)
    new = new[key.isin(C.FACTORS)]                       # 已刪除的因子不保留
    order = {k: i for i, k in enumerate(C.FACTORS)}
    key = new.index if index_cols == 1 else new.index.get_level_values(0)
    new = new.iloc[np.argsort([order.get(k, 999) for k in key], kind="stable")]
    new.to_csv(path)
    return new


def _merge_cols(new, name):
    path = _path("table", name)
    if os.path.exists(path):
        old = pd.read_csv(path, index_col=0, parse_dates=name.startswith("ic_weekly"))
        new = pd.concat([old.drop(columns=[c for c in new.columns if c in old.columns]), new], axis=1)
    new = new[[k for k in C.FACTORS if k in new.columns]]
    new.to_csv(path, date_format="%Y-%m-%d")
    return new


def main(rebuild=True, only=None):
    for d in [C.DATA_DIR, C.TABLE_DIR, C.FIG_DIR, C.LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    np.random.seed(C.SEED)
    todo = list(only) if only else list(C.FACTORS)
    partial = only is not None

    # 1. 資料
    panel, sti = dl.load_panel(rebuild=rebuild)
    days = dl.trading_days(panel)
    print(f"資料期間 {panel.index[0].date()} ~ {panel.index[-1].date()}，原始列 {len(panel)}，交易日 {len(days)}")
    print("本次計算的因子：", todo)

    # 2. 遮罩
    tradable = dl.tradable_mask(panel, days)
    tradable.to_parquet(_path("data", "tradable_mask.parquet"))

    # 3. 週頻日曆
    sched = PP.weekly_schedule(days)
    sched = sched[sched["signal"] >= C.START_SIGNAL].reset_index(drop=True)
    sched.to_csv(_path("data", "schedule.csv"), index=False, date_format="%Y-%m-%d")
    signals = pd.DatetimeIndex(sched["signal"])
    non_tue = sched[sched["rebal"].dt.day_of_week != C.REBALANCE_WEEKDAY]
    print(f"換倉次數 {len(sched)}（{sched.rebal.iloc[0].date()} ~ {sched.rebal.iloc[-1].date()}），非週二換倉 {len(non_tue)} 次")

    # 4. 因子（每日值計算很快，全部重算；部分模式下只有 todo 的值會寫回）
    fac_all = F.compute_factors(panel, sti, days)
    fac = {k: fac_all[k] for k in todo}
    fac_path = _path("data", "factors_daily.parquet")
    if partial and os.path.exists(fac_path):
        old = pd.read_parquet(fac_path)
        new = pd.concat({k: v.stack(future_stack=True) for k, v in fac.items()}, axis=1)
        new.index.names = ["date", "ticker"]
        merged = pd.concat([old.drop(columns=[c for c in todo if c in old.columns]), new], axis=1)
        merged[[k for k in C.FACTORS if k in merged.columns]].to_parquet(fac_path)
    else:
        F.save_factors(fac, fac_path)

    # 5. 預處理
    mv = dl.field(panel.loc[days], "MV")
    z_new, status = PP.preprocess_all(fac, tradable, mv, signals, factors=todo)
    z_path = _path("data", "z_factors.parquet")
    if partial and os.path.exists(z_path):
        old = pd.read_parquet(z_path)
        z = {k: old[k].unstack("ticker").reindex(index=signals, columns=C.TICKERS)
             for k in old.columns if k not in todo}
        z.update(z_new)
        z = {k: z[k] for k in C.FACTORS if k in z}
        st_path = _path("table", "preprocess_status.csv")
        if os.path.exists(st_path):
            st_old = pd.read_csv(st_path, parse_dates=["date"])
            st_old = st_old[st_old["factor"].isin(C.FACTORS) & ~st_old["factor"].isin(todo)]
            status = pd.concat([st_old, status])
    else:
        z = z_new
    PP.z_to_long(z).to_parquet(z_path)
    status.to_csv(_path("table", "preprocess_status.csv"), index=False, date_format="%Y-%m-%d")

    # 6. 持有期報酬
    rzo = PP.rz_open(panel, days)
    rz_close = dl.field(panel.loc[days], "RZ")
    fwd = {h: PP.holding_returns(rzo, sched, h, rz=rz_close) for h in C.IC_DECAY_WEEKS}
    fwd[1].stack(future_stack=True).rename("ret").to_frame().to_parquet(_path("data", "holding_returns.parquet"))

    # ---------------- 報表
    tr = tradable.reindex(signals)
    n_trad = tr.sum(axis=1).rename("n_tradable")
    n_trad.to_csv(_path("table", "tradable_count.csv"), date_format="%Y-%m-%d")
    yearly = n_trad.groupby(n_trad.index.year).agg(["min", "mean", "max"]).round(1)

    cov_rows = {}
    train = signals <= pd.Timestamp(C.TRAIN_END)
    for k in todo:
        raw = fac[k].reindex(signals).where(tr)
        cov_rows[k] = {"raw_cov_all": raw.notna().sum().sum() / tr.sum().sum(),
                       "z_cov_all": z[k].notna().sum().sum() / tr.sum().sum(),
                       "raw_cov_train": raw[train].notna().sum().sum() / tr[train].sum().sum(),
                       "z_cov_train": z[k][train].notna().sum().sum() / tr[train].sum().sum(),
                       "weeks_all_nan": int(z[k].notna().sum(axis=1).eq(0).sum())}
    coverage = _merge_rows(pd.DataFrame(cov_rows).T, "factor_coverage.csv")
    per_stock = pd.DataFrame({k: fac[k].reindex(signals).where(tr).notna().sum() / tr.sum() for k in todo})
    _merge_cols(per_stock, "factor_coverage_by_stock.csv")

    st_sum = status.groupby(["factor", "status"]).size().unstack(fill_value=0)
    st_sum = st_sum.reindex([k for k in C.FACTORS if k in st_sum.index])
    st_sum.to_csv(_path("table", "preprocess_status_summary.csv"))

    # IC 檢定（只用訓練期）；相關矩陣需要所有因子的兩兩組合，用全部 z 重算
    zt_new = {k: z[k][train] for k in todo}
    fwd_t = {h: v.reindex(signals[train]) for h, v in fwd.items()}
    summ_new, ic_weekly_new = IC.ic_summary(zt_new, fwd_t[1])
    summ = _merge_rows(summ_new, "ic_summary_train.csv")
    _merge_cols(ic_weekly_new, "ic_weekly_train.csv")
    decay = _merge_rows(IC.ic_decay(zt_new, fwd_t), "ic_decay_train.csv", index_cols=2)
    corr = IC.factor_corr({k: v[train] for k, v in z.items()})
    corr.to_csv(_path("table", "factor_corr_train.csv"))

    # 換倉日個股缺值（PO 缺或 VO 不 > 0，但訊號日可交易）
    p = panel.loc[days]
    po, vo = dl.field(p, "PO"), dl.field(p, "VO")
    reb = pd.DatetimeIndex(sched["rebal"])
    bad = (po.reindex(reb).isna() | ~(vo.reindex(reb) > 0)).values & tr.values
    miss = pd.DataFrame(bad, index=reb, columns=tr.columns)
    miss_by_stock = miss.sum()[miss.sum() > 0].sort_values(ascending=False)
    miss_by_stock.rename("n_rebal_days").to_csv(_path("table", "rebal_day_missing.csv"))

    # 資料問題摘要
    rz = dl.field(p, "RZ")
    live = rz.notna()
    pe, eps, ph, pl, px = (dl.field(p, k) for k in ["PE", "EPS", "PH", "PL", "P"])
    pos = (pe > 0) & (eps > 0)
    issues = pd.DataFrame({
        "first_date": rz.apply(lambda s: s.first_valid_index()).dt.date,
        "PO_nan_live": (po.isna() & live).sum(),
        "VO_nan_live": (vo.isna() & live).sum(),
        "VO_zero_live": ((vo == 0) & live).sum(),
        "EPS1FD12_nan_live": (dl.field(p, "EPS1FD12").isna() & live).sum(),
        "PE_nan_live": (pe.isna() & live).sum(),
        "EPS_zero": (eps == 0).sum(),
        "PE_nan_EPS_pos": (pe.isna() & (eps > 0)).sum(),
        "EPS_over_P_x_PE_median": (eps / px * pe).where(pos).median(),
        "PH_eq_PL_traded": ((ph == pl) & (vo > 0)).sum(),
        "tradable_share_after_listing": tradable.sum() / live.sum(),
        "max_abs_daily_ret": rz.pct_change(fill_method=None).abs().max(),
    })
    issues.to_csv(_path("table", "data_issues.csv"))

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\n每期可交易股票數（依年）\n", yearly)
    print("\n因子覆蓋率\n", coverage.round(3))
    print("\n預處理狀態\n", st_sum)
    print("\nIC 摘要（訓練期）\n", summ.round(4))
    print("\nIC 衰減（IC 平均）\n", decay["IC_mean"].unstack().round(4))
    print("\nIC 衰減（NW t）\n", decay["NW_t"].unstack().round(2))
    print("\n因子相關矩陣（訓練期平均截面 Spearman）\n", corr.round(2))
    print("\n換倉日個股缺值（訊號日可交易、換倉日 PO 缺或沒成交）：共", int(miss.values.sum()), "次\n", miss_by_stock)
    print("\n資料問題\n", issues)
    if not partial:
        universe_reports(panel, days, tradable, sched, fac_all, summ)
    return dict(panel=panel, sti=sti, days=days, tradable=tradable, sched=sched, fac=fac_all, z=z, fwd=fwd)


def universe_reports(panel, days, tradable, sched, fac, summ):
    """股票池相關的報表：成份股檔數與不可交易原因、成份股變動事件、下市股 L_i、
    10 檔下市股的因子覆蓋率、幣別反推、通過 IC 篩選的因子清單。"""
    signals = pd.DatetimeIndex(sched["signal"])
    rebal = pd.Series(pd.DatetimeIndex(sched["rebal"]), index=signals)
    p = panel.loc[days]
    member = dl.membership_mask(signals, "in_index")
    tr = tradable.reindex(signals).fillna(False).astype(bool)

    # ---- 每期成份股檔數、可交易檔數；不可交易的成份股和原因
    rz, mv, vo = (dl.field(p, k).reindex(signals) for k in ["RZ", "MV", "VO"])
    recent = (dl.field(p, "VO") > 0).astype(float).rolling(C.TRADABLE_LOOKBACK, min_periods=1).sum() \
        .reindex(signals) >= C.TRADABLE_MIN_DAYS
    rows = []
    for d in signals:
        for t in member.columns[member.loc[d].values & ~tr.loc[d].values]:
            if t in C.DELISTED and d > pd.Timestamp(C.DELISTED[t]):
                why = f"已在 {C.DELISTED[t]} 停止交易（L_i）"
            elif not (rz.at[d, t] == rz.at[d, t] and mv.at[d, t] > 0):
                why = "RZ 或 MV 缺值"
            elif not vo.at[d, t] > 0:
                why = "訊號日沒有成交（VO 不 > 0）"
            elif not recent.at[d, t]:
                why = "過去 20 個交易日成交不到 5 天"
            else:
                why = "其他"
            rows.append({"signal": d, "rebal": rebal[d], "ticker": t, "reason": why})
    not_trad = pd.DataFrame(rows, columns=["signal", "rebal", "ticker", "reason"])
    count = pd.DataFrame({"n_member": member.sum(axis=1), "n_tradable": tr.sum(axis=1)})
    count.index.name = "signal"
    count.to_csv(_path("table", "membership_count.csv"), date_format="%Y-%m-%d")
    not_trad.to_csv(_path("table", "member_not_tradable.csv"), index=False, date_format="%Y-%m-%d")

    # ---- 成份股變動事件：生效日之前最後一個訊號日用舊名單、生效日（含）之後第一個訊號日用新名單
    per = dl.membership_periods()
    ev = []
    for _, r in per.iterrows():
        if r["start"] > pd.Timestamp(C.START_DATA):
            ev.append({"effective": r["start"], "ticker": r["ds_ticker"], "change": "in", "est": r["start_est"]})
        if pd.notna(r["end"]):
            ev.append({"effective": r["end"] + pd.Timedelta(days=1), "ticker": r["ds_ticker"], "change": "out",
                       "est": r["end_est"]})
    ev = pd.DataFrame(ev)
    events = []
    for eff, g in ev.groupby("effective"):
        before, after = signals[signals < eff], signals[signals >= eff]
        sb = before[-1] if len(before) else pd.NaT
        sa = after[0] if len(after) else pd.NaT
        ins, outs = g.loc[g.change == "in", "ticker"].tolist(), g.loc[g.change == "out", "ticker"].tolist()
        ok_before = pd.isna(sb) or (all(member.at[sb, t] for t in outs) and not any(member.at[sb, t] for t in ins))
        ok_after = pd.isna(sa) or (all(member.at[sa, t] for t in ins) and not any(member.at[sa, t] for t in outs))
        events.append({"effective": eff.date(), "weekday": eff.day_name()[:3], "in": ",".join(ins),
                       "out": ",".join(outs), "estimated": bool(g["est"].any()),
                       "last_signal_old": sb.date() if pd.notna(sb) else None,
                       "first_signal_new": sa.date() if pd.notna(sa) else None,
                       "first_rebal_new": rebal[sa].date() if pd.notna(sa) else None,
                       "n_member_after": int(member.loc[sa].sum()) if pd.notna(sa) else None,
                       "check_ok": bool(ok_before and ok_after)})
    events = pd.DataFrame(events)
    events.to_csv(_path("table", "membership_events.csv"), index=False)

    # ---- 下市股 L_i
    last_vo = dl.last_trade_days(panel)
    dl_rows = []
    for t in C.DELISTED_SHEET:
        rr = per[per.ds_ticker == t]
        mem_sig = member.index[member[t].values]
        trd_sig = tr.index[tr[t].values]
        L = pd.Timestamp(C.DELISTED[t]) if t in C.DELISTED else pd.NaT
        dl_rows.append({"ticker": t, "name": dl.ticker_names()[t], "L_i": L.date() if pd.notna(L) else "仍在交易",
                        "last_VO_pos_data": last_vo[t].date(), "first_in_index": rr["start"].min().date(),
                        "last_in_index": rr["end"].max().date() if rr["end"].notna().all() else None,
                        "last_member_signal": mem_sig.max().date() if len(mem_sig) else None,
                        "last_tradable_signal": trd_sig.max().date() if len(trd_sig) else None,
                        "n_member_signals": len(mem_sig), "n_tradable_signals": len(trd_sig),
                        "member_after_L": int((mem_sig > L).sum()) if pd.notna(L) else 0})
    delist = pd.DataFrame(dl_rows).set_index("ticker")
    delist.to_csv(_path("table", "delisted_Li.csv"))

    # ---- 10 檔下市股的因子覆蓋率（只看它們在股票池內可交易的訊號日）
    cov = pd.DataFrame({k: fac[k].reindex(signals).where(tr).notna().sum() / tr.sum() for k in C.FACTORS})
    cov_del = cov.loc[C.DELISTED_SHEET].assign(n_tradable_signals=tr[C.DELISTED_SHEET].sum())
    cov_del.to_csv(_path("table", "factor_coverage_delisted.csv"))

    # ---- 幣別反推（2.5）：MV/(UP×NOSH)、EPS×PE/P、過去 365 天股息加總/UP ÷ DY（panel 已換匯，報價幣別一致時 ≈ 1）
    px, up, nosh, eps, pe, udd = (dl.field(p, k) for k in ["P", "UP", "NOSH", "EPS", "PE", "UDD"])
    mvd, dyld = dl.field(p, "MV"), dl.field(p, "DY")
    div365 = udd.fillna(0).rolling("365D").sum()
    traded = dl.field(p, "VO") > 0
    ccy = pd.DataFrame({
        "MV_over_UPxNOSH": (mvd * 1000 / (up * nosh)).where(traded).median(),
        "EPSxPE_over_P": (eps * pe / px).where((pe > 0) & (eps > 0)).median(),
        "DIV365_over_UP_over_DY": (div365 / up / (dyld / 100)).where(traded & (dyld > 0) & (div365 > 0))
        .loc[pd.Timestamp(C.START_DATA) + pd.Timedelta(days=365):].median(),
        "price_ccy": pd.Series({t: "USD→SGD" if t in C.USD_QUOTED else "SGD" for t in C.TICKERS}),
        "EPS_ccy": pd.Series({t: C.EPS_CCY.get(t, "SGD") for t in C.TICKERS}),
        "DIV_ccy": pd.Series({t: C.DIV_CCY.get(t, "SGD") for t in C.TICKERS}),
    })
    ccy.to_csv(_path("table", "currency_check.csv"))

    # ---- 篩選清單（訓練期 NW t > SCREEN_T_MIN）
    cmp = pd.DataFrame({"IC_mean": summ["IC_mean"], "NW_t": summ["NW_t"]})
    cmp["pass_screen"] = cmp["NW_t"] > C.SCREEN_T_MIN
    cmp["direction"] = pd.Series(C.FACTOR_SIGN)

    print("\n===== 股票池（每期 STI 成份股）=====")
    print(f"每期成份股檔數：{count['n_member'].min()}–{count['n_member'].max()}；不是 30 的訊號日 "
          f"{int((count['n_member'] != 30).sum())} 個")
    print(f"可交易檔數不是 30 的訊號日 {int((count['n_tradable'] != 30).sum())} 個；不可交易的原因：\n",
          not_trad.groupby(["reason"]).size())
    print("\n不可交易的成份股（依股票）：\n", not_trad.groupby(["ticker", "reason"]).size())
    print("\n成份股變動事件（共 %d 次，檢查通過 %d 次）\n" % (len(events), events["check_ok"].sum()),
          events.to_string(index=False))
    print("\n下市股\n", delist.to_string())
    print("\n10 檔下市股的因子覆蓋率\n", cov_del.round(2).to_string())
    print("\n幣別反推（10 檔下市股 + 外幣相關股票）\n",
          ccy.loc[[t for t in C.TICKERS if t in C.DELISTED_SHEET or t in C.EPS_CCY or t in C.USD_QUOTED]].round(3))
    print("\n通過篩選（NW t > %.0f）：" % C.SCREEN_T_MIN, cmp.index[cmp["pass_screen"]].tolist())
    return dict(count=count, not_trad=not_trad, events=events, delist=delist, cov_del=cov_del, ccy=ccy, cmp=cmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="'new' = config.NEW_FACTORS，或以逗號分隔的因子代號")
    ap.add_argument("--no-rebuild", action="store_true", help="直接讀 outputs/data 的 parquet，不重讀 Excel")
    a = ap.parse_args()
    only = None if a.only is None else (C.NEW_FACTORS if a.only == "new" else a.only.split(","))
    main(rebuild=not a.no_rebuild, only=only)
