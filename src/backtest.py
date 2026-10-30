"""回測引擎：每日淨值、換倉日開盤成交、權重漂移、交易成本、交易紀錄。"""
import copy

import numpy as np
import pandas as pd

import config as C
import portfolio as P


class Market:
    """回測需要的日資料（交易日 × 股票），全部是 ndarray。"""

    def __init__(self, days, rz, po, px, vo, tickers, delisted=None):
        self.days = pd.DatetimeIndex(days)
        self.tickers = list(tickers)
        # 下市（2.7）：{交易日位置: [股票位置]}，持有中的部位在 L_i 收盤轉成現金（不扣成本）
        self.delist_at = {}
        for t, d in (C.DELISTED if delisted is None else delisted).items():
            d = pd.Timestamp(d)
            if t in self.tickers and d in self.days:
                self.delist_at.setdefault(self.days.get_loc(d), []).append(self.tickers.index(t))
        self.rz = rz
        with np.errstate(invalid="ignore", divide="ignore"):
            self.rzo = rz * po / np.where(px > 0, px, np.nan)
            prev = np.vstack([np.full((1, rz.shape[1]), np.nan), rz[:-1]])
            self.r_cc = np.where(np.isfinite(rz / prev), rz / prev - 1, 0.0)
            on = self.rzo / prev - 1
            idr = rz / self.rzo - 1
        self._traded = vo > 0
        self.exec_ok = np.isfinite(po) & (vo > 0)            # 換倉日開盤可以成交
        has_open = np.isfinite(on) & np.isfinite(idr) & self.exec_ok
        # 開盤不能成交的股票（停牌）：整天的報酬放在「前一日收盤 → 開盤」那一段，權重不變
        self.r_on = np.where(has_open, on, self.r_cc)
        self.r_id = np.where(has_open, idr, 0.0)
        self.pos = {d: i for i, d in enumerate(self.days)}

    def close_execution(self):
        """敏感度測試「週二收盤換倉」：換倉日整天用舊權重（收盤 → 收盤），收盤時換倉；
        成交價和交易損益都用收盤的 RZ；當天有成交（VO > 0）才能交易。"""
        m = copy.copy(self)
        m.r_on = self.r_cc
        m.r_id = np.zeros_like(self.r_cc)
        m.rzo = self.rz
        m.exec_ok = self._traded & np.isfinite(self.rz)
        return m

    @classmethod
    def from_panel(cls, panel, days):
        import data_loader as dl
        p = panel.loc[days]
        f = lambda k: dl.field(p, k).values.astype(float)
        return cls(days, f("RZ"), f("PO"), f("P"), np.nan_to_num(f("VO"), nan=0.0), C.TICKERS)


def run(mkt, rebal_dates, rel_targets, exposure=None, cost_bps=C.COST_BPS, band=C.NO_TRADE_BAND,
        stock_cap=C.MAX_WEIGHT, sector=None, sector_caps_rel=None, end_date=None, sigma=None, priority=None):
    """rebal_dates[k] 開盤換到目標 exposure[k] × rel_targets[k]（rel 加總 1）。
    sector_caps_rel[k]：各產業占股票部位的上限（inf = 不設）。
    priority[k]：選股組當期的分數（賣不掉時決定少買哪一檔），非選股組為 None。回傳 dict。"""
    N = len(mkt.tickers)
    K = len(rebal_dates)
    exposure = np.ones(K) if exposure is None else np.asarray(exposure, float)
    reb_pos = [mkt.pos[pd.Timestamp(d)] for d in rebal_dates]
    first = reb_pos[0]
    last = mkt.pos[pd.Timestamp(end_date)] if end_date is not None else len(mkt.days) - 1
    reb_at = {p: k for k, p in enumerate(reb_pos) if p <= last}
    w = np.zeros(N); cash = 1.0; nav = 1.0
    days_out, nav_out, net_out, gross_out, cash_out, cost_out = [], [], [], [], [], []
    w_rec, rec = [], []
    open_trades, trades = {}, []
    c = cost_bps / 1e4
    for t in range(first, last + 1):
        if t in reb_at:
            k = reb_at[t]
            g_on = w @ mkt.r_on[t]
            nav_open = nav * (1 + g_on)
            w = w * (1 + mkt.r_on[t]) / (1 + g_on); cash = cash / (1 + g_on)
            w0, cash0 = w.copy(), cash
            e = float(np.clip(exposure[k], 0, 1)) if np.isfinite(exposure[k]) else 1.0
            target = e * rel_targets[k]
            new, info = _rebalance(w0, target, mkt.exec_ok[t], e, band, stock_cap, sector,
                                   None if sector_caps_rel is None else sector_caps_rel[k],
                                   None if priority is None else priority[k])
            cash_new = 1 - new.sum()
            turnover = np.abs(new - w0).sum() + abs(cash_new - cash0)
            cost = turnover * c
            nav_after = nav_open * (1 - cost)
            g_id = new @ mkt.r_id[t]
            nav_close = nav_after * (1 + g_id)
            # 交易紀錄
            for i in np.where((w0 <= 0) & (new > 0))[0]:
                open_trades[i] = (t, new[i])
            for i in np.where((w0 > 0) & (new <= 0))[0]:
                if i in open_trades:
                    t0, wb = open_trades.pop(i)
                    trades.append(_trade(mkt, i, t0, t, wb, closed=True))
            w = new * (1 + mkt.r_id[t]) / (1 + g_id); cash = cash_new / (1 + g_id)
            day_ret = nav_close / nav - 1
            gross = (1 + g_on) * (1 + g_id) - 1
            stock_w = new.sum()
            rel = new / stock_w if stock_w > 0 else new
            rec.append({"rebal": mkt.days[t], "exposure_target": e, "stock_weight": stock_w, "cash": cash_new,
                        "turnover": turnover, "one_way_turnover": np.abs(new - w0).sum() / 2, "cost": cost,
                        "n_hold": int((new > 0).sum()), "max_weight": new.max() if N else 0,
                        "hhi": float((rel ** 2).sum()), "n_frozen": info["n_frozen"], "n_banded": info["n_banded"],
                        "n_blocked": info["n_blocked"], "n_skipped_buy": info["n_skipped_buy"],
                        "cap_status": info["cap_status"],
                        "frozen_over_cap": info["frozen_over_cap"],
                        "min_sigma": float(np.nanmin(np.where(new > 0, sigma[k], np.nan))) if sigma is not None and (new > 0).any() else np.nan})
            w_rec.append(new.copy())
            cost_out.append(cost)
            nav = nav_close
        else:
            g = w @ mkt.r_cc[t]
            day_ret = g
            gross = g
            nav = nav * (1 + g)
            w = w * (1 + mkt.r_cc[t]) / (1 + g); cash = cash / (1 + g)
            cost_out.append(0.0)
        for i in mkt.delist_at.get(t, ()):                    # 下市：L_i 收盤的部位價值轉成現金，淨值不變
            if w[i] > 0:
                cash += w[i]; w[i] = 0.0
                if i in open_trades:
                    t0, wb = open_trades.pop(i)
                    trades.append(_trade(mkt, i, t0, t, wb, closed=True, delisted=True))
        days_out.append(mkt.days[t]); nav_out.append(nav); net_out.append(day_ret)
        gross_out.append(gross); cash_out.append(cash)
    for i, (t0, wb) in open_trades.items():
        trades.append(_trade(mkt, i, t0, last, wb, closed=False))
    daily = pd.DataFrame({"nav": nav_out, "ret": net_out, "ret_gross": gross_out, "cash": cash_out,
                          "cost": cost_out}, index=pd.DatetimeIndex(days_out))
    reb = pd.DataFrame(rec).set_index("rebal")
    weights = pd.DataFrame(w_rec, index=reb.index, columns=mkt.tickers)
    return {"daily": daily, "rebal": reb, "weights": weights, "trades": pd.DataFrame(trades)}


def _trade(mkt, i, t0, t1, wb, closed, delisted=False):
    if delisted:                                              # 在 L_i 收盤轉現金
        pnl = mkt.rz[t1, i] / mkt.rzo[t0, i] - 1
    elif closed:
        pnl = mkt.rzo[t1, i] / mkt.rzo[t0, i] - 1
    else:
        pnl = mkt.rz[t1, i] / mkt.rzo[t0, i] - 1
    return {"ticker": mkt.tickers[i], "buy_date": mkt.days[t0], "sell_date": mkt.days[t1] if closed else pd.NaT,
            "buy_weight": wb, "hold_days": t1 - t0, "pnl": pnl, "closed": closed, "delisted": delisted}


def _rebalance(w0, target, exec_ok, total, band, stock_cap, sector, sec_caps_rel, priority=None):
    """把目標換成可執行的新權重（占淨值）。
    - 開盤不能成交的股票：已持有 → 維持漂移後權重（frozen，固定不動）；沒持有 → 不得買進（blocked）。
    - 選股組（priority = 當期分數）：要賣卻賣不掉的持股有幾檔，就少買幾檔新股（去掉分數最低的新買進），
      讓持股數維持 K。
    - 不交易區間：繼續持有的股票（目前 > 0、目標 > 0），目標和目前相差 < band → 改回目前權重，
      然後所有可成交的股票一起重新正規化到 total − 停牌部位，再套一次上限。"""
    n = len(w0)
    frozen = ~exec_ok & (w0 > 0)
    blocked = ~exec_ok & (w0 <= 0) & (target > 0)
    t = np.where(blocked, 0.0, target)
    n_skipped = 0
    if priority is not None:
        stuck = frozen & (target <= 0)
        buys = np.where(exec_ok & (w0 <= 0) & (t > 0))[0]
        n_skipped = min(int(stuck.sum()), len(buys))
        if n_skipped:
            pr = np.where(np.isfinite(priority[buys]), priority[buys], -np.inf)
            t[buys[np.argsort(pr, kind="stable")[:n_skipped]]] = 0.0
    banded = np.zeros(n, bool)
    if band is not None:
        banded = exec_ok & (w0 > 0) & (t > 0) & (np.abs(t - w0) < band)
    new = np.where(frozen | banded, w0, t)
    free = ~frozen & (new > 0)
    room = total - w0[frozen].sum()
    status = "ok"
    if free.any() and room > 0:
        new[free] *= room / new[free].sum()
        if stock_cap is not None or sec_caps_rel is not None:
            caps = np.full(int(sector.max()) + 1, np.inf) if sec_caps_rel is None else sec_caps_rel
            new, status = P.apply_caps(new, sector, stock_cap, caps, fixed=~free, total=total)
    elif free.any():
        new[free] = 0.0
    new = np.clip(new, 0, None)
    cap = np.inf if stock_cap is None else stock_cap
    return new, {"n_frozen": int(frozen.sum()), "n_banded": int(banded.sum()), "n_blocked": int(blocked.sum()),
                 "n_skipped_buy": n_skipped, "cap_status": status, "frozen_over_cap": int((frozen & (w0 > cap + 1e-10)).sum())}


def run_index(days, ret, start, end):
    """B0：買進持有指數（日報酬 ret，Series）。"""
    r = ret.loc[start:end].copy()
    r.iloc[0] = 0.0 if not np.isfinite(r.iloc[0]) else r.iloc[0]
    r = r.fillna(0.0)
    daily = pd.DataFrame({"ret": r, "ret_gross": r, "cash": 0.0, "cost": 0.0})
    daily["nav"] = (1 + r).cumprod()
    return {"daily": daily, "rebal": pd.DataFrame(), "weights": pd.DataFrame(), "trades": pd.DataFrame()}
