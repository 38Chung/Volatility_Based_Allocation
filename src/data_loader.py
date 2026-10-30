"""讀取股票與 STI 日資料、匯率、成份股期間；建立交易日與可交易遮罩。"""
import os

import numpy as np
import pandas as pd

import config as C


# ---------------------------------------------------------------- 讀檔
def read_raw():
    """讀 data/All_Stock_STI_used.csv（長表：date、ticker + 18 個欄位，Datastream 原始值、原始幣別）
    和 data/STI_used.csv。回傳 (stocks, sti)：stocks 是 日期 × (ticker, field) 的寬表，只含 config.TICKERS × config.FIELDS。
    CSV 是從 Datastream 的 Excel（DSGRID 格式）原樣取出：尚未換匯、下市股 L_i 之後的補值也還在（由 build_panel 處理）。"""
    raw = pd.read_csv(C.DATA_FILE, parse_dates=["date"])
    stocks = raw.set_index(["date", "ticker"])[C.FIELDS].unstack("ticker").swaplevel(axis=1)
    stocks.columns.names = ["ticker", "field"]
    missing = [(t, f) for t in C.TICKERS for f in C.FIELDS if (t, f) not in stocks.columns]
    assert not missing, f"缺少欄位：{missing[:5]}"
    stocks = stocks.loc[:, pd.MultiIndex.from_product([C.TICKERS, C.FIELDS], names=["ticker", "field"])]
    sti = pd.read_csv(C.STI_FILE, index_col="date", parse_dates=True)[["PI", "RI", "DY", "MV"]]
    idx = stocks.index.union(sti.index).sort_values()
    return stocks.reindex(idx).astype(float), sti.reindex(idx).astype(float)


def last_trade_days(stocks):
    """每檔股票全樣本最後一個 VO > 0 的日子。"""
    vo = stocks.xs("VO", axis=1, level="field")
    return vo.apply(lambda s: s[s > 0].last_valid_index())


def check_delisted(stocks):
    """用資料重算 config.DELISTED 的最後真實交易日，回傳比對表。"""
    last = last_trade_days(stocks)
    rows = []
    for t in C.DELISTED_SHEET:
        cfg = pd.Timestamp(C.DELISTED[t]) if t in C.DELISTED else pd.NaT
        rows.append({"ticker": t, "L_config": cfg, "last_VO_pos": last[t], "match": cfg == last[t] if t in C.DELISTED
                     else last[t] >= stocks.index[-1] - pd.Timedelta(days=7)})
    return pd.DataFrame(rows).set_index("ticker")


def mask_after_delisting(stocks):
    """config.DELISTED 的股票：L_i 之後所有欄位設成 NaN（Datastream 用最後價格往後補值，不能用）。"""
    stocks = stocks.copy()
    for t, d in C.DELISTED.items():
        if t in stocks.columns.get_level_values("ticker"):
            stocks.loc[stocks.index > pd.Timestamp(d), t] = np.nan
    return stocks


def _panel_path():
    return os.path.join(C.DATA_DIR, "panel_daily.parquet")


def _sti_path():
    return os.path.join(C.DATA_DIR, "sti_daily.parquet")


def build_panel():
    """從 data/ 的 CSV 讀取並存成 parquet。
    最後一個「至少一檔 VO > 0」的日子之後的列刪掉（10/30 是空列）；下市股 L_i 之後設 NaN；美元報價股換成新元。"""
    os.makedirs(C.DATA_DIR, exist_ok=True)
    stocks, sti = read_raw()
    vo = stocks.xs("VO", axis=1, level="field")
    last_day = vo.index[(vo > 0).any(axis=1)][-1]
    stocks, sti = stocks.loc[:last_day], sti.loc[:last_day]
    chk = check_delisted(stocks)
    chk.to_csv(os.path.join(C.DATA_DIR, "delisted_check.csv"), date_format="%Y-%m-%d")
    if not chk["match"].all():
        print("警告：最後真實交易日跟 config.DELISTED 不一致：\n", chk[~chk["match"]])
    stocks = to_sgd(mask_after_delisting(stocks))
    stocks.to_parquet(_panel_path())
    sti.to_parquet(_sti_path())
    return stocks, sti


def load_panel(rebuild=False):
    """回傳 (panel, sti)。panel 欄位為 MultiIndex (ticker, field)，列為原始日期（含假日）。"""
    if rebuild or not (os.path.exists(_panel_path()) and os.path.exists(_sti_path())):
        return build_panel()
    stocks = pd.read_parquet(_panel_path())
    stocks.columns = pd.MultiIndex.from_tuples(stocks.columns, names=["ticker", "field"])
    return stocks, pd.read_parquet(_sti_path())


def field(panel, name):
    """取出一個欄位：日期 × 股票。"""
    return panel.xs(name, axis=1, level="field")[C.TICKERS]


def ticker_names():
    names = pd.read_csv(os.path.join(C.SRC_DATA_DIR, "tickers.csv"), index_col=0)["name"]
    return names.reindex(C.TICKERS)


# ---------------------------------------------------------------- 匯率
def mas_fx_daily():
    """MAS 每日中午匯率（data_extra/mas_fx_daily.csv，由 fetch_mas_fx.py 下載）：1 單位外幣 = 多少新元。"""
    return pd.read_csv(os.path.join(C.EXTRA_DIR, "mas_fx_daily.csv"), index_col=0, parse_dates=True).sort_index()


def _rate_asof(dates, currency, lag_days):
    """每個日期取「日期 − lag_days 天」當天或之前最近一個 MAS 匯率；資料開始前用第一筆。"""
    rates = mas_fx_daily()[currency].dropna()
    dates = pd.DatetimeIndex(dates)
    pos = rates.index.searchsorted(dates - pd.Timedelta(days=lag_days), side="right") - 1
    vals = rates.values[np.clip(pos, 0, None)]
    return pd.Series(vals, index=dates)


def rate_by_date(dates, currency):
    """因子換匯（EPS1FD12、UDD）：用前一天（含）以前最近一個每日匯率，保守作法（中午匯率其實在收盤前就知道）。"""
    return _rate_asof(dates, currency, lag_days=1)


def to_sgd(stocks):
    """美元報價股的價格類欄位換成新元。收盤類（RZ、P、UP、PH、PL、MV、VA）用當天匯率；
    開盤價 PO 用前一天的匯率（開盤 9:00 時當天中午的匯率還不知道）。"""
    stocks = stocks.copy()
    same_day = _rate_asof(stocks.index, "USD", lag_days=0).values
    prev_day = _rate_asof(stocks.index, "USD", lag_days=1).values
    for t in C.USD_QUOTED:
        for f in C.PRICE_FIELDS:
            stocks[(t, f)] = stocks[(t, f)] * (prev_day if f == "PO" else same_day)
    return stocks


# ---------------------------------------------------------------- 交易日與遮罩
def trading_days(panel):
    """至少一檔股票 VO > 0 的日子。"""
    vo = field(panel, "VO")
    return vo.index[(vo > 0).any(axis=1)]


def tradable_mask(panel, days=None, membership=C.MEMBERSHIP_MASK, exclude=()):
    """每個交易日的可交易遮罩：RZ、MV 有值且 MV > 0；當天 VO > 0；過去 20 個交易日中至少 5 天 VO > 0；
    並且在成份股遮罩內。下市股 L_i 之後的值在讀檔時已設成 NaN，所以「日期 ≤ L_i」由 RZ 有值自動滿足。
    membership：None / "ever_included" / "in_index"（主設定）；exclude：整段期間都不納入的股票（敏感度測試用）。"""
    if days is None:
        days = trading_days(panel)
    p = panel.loc[days]
    rz, mv, vo = field(p, "RZ"), field(p, "MV"), field(p, "VO")
    traded = (vo > 0)
    recent = traded.astype(float).rolling(C.TRADABLE_LOOKBACK, min_periods=1).sum() >= C.TRADABLE_MIN_DAYS
    mask = rz.notna() & mv.notna() & (mv > 0) & traded & recent
    mask = apply_membership(mask, membership)
    for t in exclude:
        mask[t] = False
    return mask


def membership_periods():
    """讀 data/STI_membership_used.csv（STI 成份期間）：回傳 DataFrame(ds_ticker, sgx, name, start, end, start_est, end_est)。
    end 為空表示至今仍在指數；start_est / end_est = 說明欄寫「估計」或「推定」（日期是推定值）。"""
    raw = pd.read_csv(C.MEMBERSHIP_FILE, dtype=str)
    raw = raw.rename(columns={"sgx_code": "sgx"})
    raw["start"] = pd.to_datetime(raw["start"])
    raw["end"] = pd.to_datetime(raw["end"], errors="coerce")
    est = lambda s: s.fillna("").astype(str).str.contains("估計|推定")
    raw["start_est"], raw["end_est"] = est(raw["start_note"]), est(raw["end_note"])
    sgx_to_ds = {v: k for k, v in C.DS_TO_SGX.items()}
    raw["ds_ticker"] = raw["sgx"].astype(str).str.strip().map(sgx_to_ds)
    unknown = raw.loc[raw["ds_ticker"].isna(), "sgx"].tolist()
    assert not unknown, f"成份期間裡有對不到 Datastream 代碼的 SGX 代號：{unknown}"
    cols = ["ds_ticker", "sgx", "name", "start", "end", "start_est", "end_est", "start_note", "end_note"]
    return raw[cols].reset_index(drop=True)


def membership_mask(dates, mode):
    """mode = "ever_included"：日期 ≥ 第一次進入日；"in_index"：日期落在某一段 [進入日, 最後在指數日]。"""
    per = membership_periods()
    out = pd.DataFrame(False, index=dates, columns=C.TICKERS)
    for t in C.TICKERS:
        rows = per[per["ds_ticker"] == t]
        if mode == "ever_included":
            if len(rows):
                out[t] = dates >= rows["start"].min()
        elif mode == "in_index":
            m = np.zeros(len(dates), dtype=bool)
            for _, r in rows.iterrows():
                end = r["end"] if pd.notna(r["end"]) else pd.Timestamp.max
                m |= (dates >= r["start"]) & (dates <= end)
            out[t] = m
        else:
            raise ValueError(mode)
    return out


def apply_membership(mask, mode):
    if mode is None:
        return mask
    return mask & membership_mask(mask.index, mode)
