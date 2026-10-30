"""從 MAS 網站下載每日匯率（新元 / 外幣；MAS 註明每日值是中午的匯率），存成 data_extra/mas_fx_daily.csv。

欄位：USD（S$ / 1 USD）、CNY、HKD、MYR、PHP、THB（原始為每 100 單位，這裡已除以 100，都是 S$ / 1 單位）。
來源：https://eservices.mas.gov.sg/statistics/msb/exchangerates.aspx（Exchange Rates, Daily；網頁欄名是 End of Period，但註腳寫 daily figures are values as of noon）
用法：python fetch_mas_fx.py [起始年] [結束年]
"""
import http.cookiejar
import io
import os
import re
import sys
import urllib.parse
import urllib.request

import pandas as pd

import config as C

URL = "https://eservices.mas.gov.sg/statistics/msb/exchangerates.aspx"
P = "ctl00$ContentPlaceHolder1$"
# 勾選的幣別：每 1 單位清單的第 2 項 = USD；每 100 單位清單的第 2 項 = CNY、第 3 項 = HKD、第 8 項 = MYR、第 11 項 = PHP、第 15 項 = THB
SELECT = {P + "EndOfPeriodPerUnitCheckBoxList$2": "on",
          P + "EndOfPeriodPer100UnitsCheckBoxList$2": "on",
          P + "EndOfPeriodPer100UnitsCheckBoxList$3": "on",
          P + "EndOfPeriodPer100UnitsCheckBoxList$8": "on",
          P + "EndOfPeriodPer100UnitsCheckBoxList$11": "on",
          P + "EndOfPeriodPer100UnitsCheckBoxList$15": "on"}
COLS = {"S$ Per Unit of US Dollar": ("USD", 1), "S$ Per 100 Units of Chinese Renminbi": ("CNY", 100),
        "S$ Per 100 Units of Hong Kong Dollar": ("HKD", 100),
        "S$ Per 100 Units of Malaysian Ringgit": ("MYR", 100),
        "S$ Per 100 Units of Philippine Peso": ("PHP", 100),
        "S$ Per 100 Units of Thai Baht": ("THB", 100)}
MONTHS = {m: i + 1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _hidden(html):
    return dict(re.findall(r'<input type="hidden" name="([^"]+)" id="[^"]*" value="([^"]*)"', html))


def download_year(year):
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    op.addheaders = [("User-Agent", "Mozilla/5.0")]
    post = lambda html, extra: op.open(URL, urllib.parse.urlencode({**_hidden(html), **extra}).encode(), timeout=120)
    form = {P + "StartYearDropDownList": str(year), P + "EndYearDropDownList": str(year),
            P + "StartMonthDropDownList": "1", P + "EndMonthDropDownList": "12",
            P + "FrequencyDropDownList": "D"}
    html = op.open(URL, timeout=120).read().decode()
    html = post(html, {**form, "__EVENTTARGET": P + "FrequencyDropDownList"}).read().decode()   # 先切到每日
    return post(html, {**form, **SELECT, P + "DownloadButton": "Download"}).read().decode("utf-8-sig")


def parse_csv(text):
    """MAS 的 CSV：年、月只寫在該段第一列，往下補齊。"""
    lines = text.splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("End of Period"))
    df = pd.read_csv(io.StringIO("\n".join(lines[head:])), header=0, dtype=str)
    df.columns = ["year", "month", "day"] + list(df.columns[3:])
    df = df[df["day"].fillna("").str.strip().str.isdigit()].copy()
    df["year"] = df["year"].replace("", None).ffill()
    df["month"] = df["month"].replace("", None).ffill()
    df.index = pd.to_datetime(dict(year=df["year"].astype(int), month=df["month"].str.strip().map(MONTHS),
                                   day=df["day"].astype(int)))
    out = pd.DataFrame(index=df.index)
    for col, (name, unit) in COLS.items():
        out[name] = pd.to_numeric(df[col], errors="coerce") / unit
    return out


def main(start=2010, end=2026):
    frames = []
    for y in range(start, end + 1):
        fx = parse_csv(download_year(y))
        print(f"{y}: {len(fx)} 天，{fx.index.min().date()} ~ {fx.index.max().date()}")
        frames.append(fx)
    fx = pd.concat(frames).sort_index()
    fx = fx[~fx.index.duplicated()]
    os.makedirs(C.EXTRA_DIR, exist_ok=True)
    path = os.path.join(C.EXTRA_DIR, "mas_fx_daily.csv")
    fx.index.name = "date"
    fx.to_csv(path)
    print(f"存檔 {path}：{len(fx)} 天，缺值 {fx.isna().sum().to_dict()}")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:3]))
