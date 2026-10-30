"""所有日期、參數、路徑。參數預設值在這裡；凍結後實際使用的值在 outputs/frozen_params.json（以凍結檔為準）。"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DATA_DIR = os.path.join(ROOT, "data")       # 輸入資料（見 README）
DATA_FILE = os.path.join(SRC_DATA_DIR, "All_Stock_STI_used.csv")      # 51 檔股票的日資料（原始幣別），長表
STI_FILE = os.path.join(SRC_DATA_DIR, "STI_used.csv")                  # STI：PI、RI、DY、MV
MEMBERSHIP_FILE = os.path.join(SRC_DATA_DIR, "STI_membership_used.csv")  # STI 成份期間
EXTRA_DIR = SRC_DATA_DIR                                                # MAS 每日匯率 mas_fx_daily.csv
OUT_DIR = os.path.join(ROOT, "outputs")
DATA_DIR = os.path.join(OUT_DIR, "data")        # 中間檔（parquet），可重建
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
LOG_DIR = os.path.join(OUT_DIR, "logs")

# 日期
START_DATA = "2010-01-01"; START_SIGNAL = "2011-01-01"
TRAIN_END = "2022-12-31"; VALID_START = "2023-01-01"; VALID_END = "2026-07-31"
FREEZE_DATE = "2026-09-30"; TEST_START = "2026-10-01"; TEST_END = "2026-10-29"

# 模型與組合參數
LASSO_WINDOW = 260; LASSO_MIN_WINDOW = 156; LAMBDA_VALID_WEEKS = 104
# 選 λ：λ = c × λ_max、c 排除 1、驗證段簡化 S1 扣成本後 Sharpe、並列取中間點（見 src/model.py）
LAMBDA_SIM_K = 12; LAMBDA_SIM_BUFFER = 3   # 選 λ 時模擬的簡化 S1（固定 K = 12，不隨 K 的選項變）
SCREEN_T_MIN = 2.0   # 進 Lasso 前的 IC 篩選門檻：週 Rank IC 的 NW t 值 > 2
SCREEN_MODE = "static"   # "static"：訓練期（START_SIGNAL–TRAIN_END）IC 表固定篩選；"rolling"：每期用 Lasso 訓練窗口；None：不篩選
K = 12; RANK_BUFFER = 3; MAX_WEIGHT = 0.15; SECTOR_OVER = 0.10
NO_TRADE_BAND = 0.02; EWMA_LAMBDA = 0.94; SIGMA_FLOOR_PCT = 5
VOL_TARGET = 0.12; MAX_EXPOSURE = 1.0; GARCH_LOOKBACK = 500
COST_BPS = 15; USE_BM = False; MEMBERSHIP_MASK = "in_index"   # 每期成份股；None = 固定股票池（只用在「固定 38 檔」敏感度測試）
REBALANCE_WEEKDAY = 1   # 0=週一, 1=週二；遇非交易日順延到同週下一個交易日

SEED = 5210

# 股票池
SECTOR = {
    "DBSS": "Financials", "OCBC": "Financials", "UOBS": "Financials", "SISE": "Financials", "YAFH": "Financials",
    "ASCE": "REIT", "CAPI": "REIT", "FRCT": "REIT", "FRAL": "REIT", "KEDC": "REIT", "MAPI": "REIT", "MPLT": "REIT",
    "MTRE": "REIT", "CAPT": "REIT",
    "CAP1": "Real Estate", "CITY": "Real Estate", "UL@V": "Real Estate", "HKLH": "Real Estate",
    "DBSL": "Real Estate", "CMA": "Real Estate", "MGLP": "Real Estate",
    "KEPP": "Industrials", "JSHP": "Industrials", "SEMB": "Industrials", "STEN": "Industrials", "YSHL": "Industrials",
    "CODE": "Industrials", "SATS": "Industrials", "SIAE": "Industrials", "SAIR": "Industrials",
    "JMTS": "Industrials", "CYCL": "Industrials", "HPHH": "Industrials",
    "COSC": "Industrials", "SMTC": "Industrials", "NOLS": "Industrials", "NOBE": "Industrials", "JSTG": "Industrials",
    "THBE": "Consumer", "EMPE": "Consumer", "FRAS": "Consumer", "GLDA": "Consumer", "EZYH": "Consumer",
    "GINT": "Consumer", "DAIR": "Consumer", "OLAM": "Consumer",
    "TELC": "TMT & Health", "STAH": "TMT & Health", "VEMA": "TMT & Health", "IHHH": "TMT & Health",
    "SGPS": "TMT & Health",
}
# 11 檔已下市或已離開的前成份股（產業是依公司業務推定的，見 README）
DELISTED_SHEET = ["COSC", "SMTC", "NOLS", "CMA", "OLAM", "NOBE", "CAPT", "JSTG", "DBSL", "MGLP", "SGPS"]
# 最後真實交易日 L_i（全樣本最後一個 VO > 0 的日子）；之後的值是 Datastream 補值，讀檔時設成 NaN（2.7）。
# COSC 仍在交易，不在表中。data_loader 會用資料重算並比對
DELISTED = {"SMTC": "2016-10-18", "NOLS": "2016-07-18", "CMA": "2014-06-09", "OLAM": "2022-03-09",
            "NOBE": "2018-11-16", "CAPT": "2020-10-16", "JSTG": "2021-04-12", "DBSL": "2021-09-09",
            "MGLP": "2018-01-04", "SGPS": "2022-04-07"}
# 主要上市地不在 SGX 的第二上市股：IHHH（馬來西亞）、EMPE（菲律賓）。只在成份期間內納入，
# 流動性交給可交易遮罩判斷。敏感度測試「剔除 IHHH/EMPE」用這份清單
SECONDARY_LISTED = ["IHHH", "EMPE"]
EXCLUDED = []
TICKERS = sorted(t for t in SECTOR if t not in EXCLUDED)
# 固定 38 檔（今天仍上市、曾為成份股的公司）：存活者偏誤敏感度測試用（MEMBERSHIP_MASK = None，剔除第二上市股和 11 檔下市股）
FIXED38 = sorted(t for t in SECTOR if t not in SECONDARY_LISTED and t not in DELISTED_SHEET)

# 幣別（MAS 每日匯率，見 data_loader.to_sgd）
# 價格類欄位以美元報價的股票：讀檔時用當天匯率換成新元（PO 用前一天）
USD_QUOTED = ["HKLH", "JMTS", "DAIR", "HPHH", "JSTG"]   # CYCL、NOBE 在 SGX 以新元報價，只有 EPS（和股息）是美元
PRICE_FIELDS = ["RZ", "P", "UP", "PO", "PH", "PL", "MV", "VA"]
# 預測 EPS（EPS1FD12）的記錄幣別 → fep、rev
_EPS_CCY_ALL = {"THBE": "THB", "EMPE": "PHP", "IHHH": "MYR", "YSHL": "CNY",
                "HKLH": "USD", "JMTS": "USD", "DAIR": "USD", "CYCL": "USD", "HPHH": "HKD",
                "JSTG": "USD", "NOBE": "USD"}
# 股息（UDD）的記錄幣別 → dy（NOBE 是依公司以美元編製財報推定的）
_DIV_CCY_ALL = {"THBE": "THB", "EMPE": "PHP", "IHHH": "MYR",
                "HKLH": "USD", "JMTS": "USD", "DAIR": "USD", "CYCL": "USD", "HPHH": "USD",
                "JSTG": "USD", "NOBE": "USD"}
EPS_CCY = {t: c for t, c in _EPS_CCY_ALL.items() if t in TICKERS}
DIV_CCY = {t: c for t, c in _DIV_CCY_ALL.items() if t in TICKERS}

DS_TO_SGX = {"ASCE": "A17U", "CAPI": "C38U", "CAP1": "9CI", "CITY": "C09", "DBSS": "D05", "FRCT": "J69U",
             "FRAL": "BUOU", "GINT": "G13", "KEPP": "BN4", "KEDC": "AJBU", "MAPI": "ME8U", "MPLT": "M44U",
             "MTRE": "N2IU", "OCBC": "O39", "SATS": "S58", "STEN": "S63", "TELC": "Z74", "THBE": "Y92",
             "UOBS": "U11", "UL@V": "U14", "VEMA": "V03", "EZYH": "F34", "YSHL": "BS6", "JSHP": "5E2",
             "SEMB": "U96", "SAIR": "C6L", "SISE": "S68", "CODE": "C52", "EMPE": "EMI", "FRAS": "F99",
             "GLDA": "E5H", "IHHH": "Q0F", "SIAE": "S59", "STAH": "CC3", "YAFH": "YF8",
             "HKLH": "H78", "JMTS": "J36", "DAIR": "D01", "CYCL": "C07", "HPHH": "NS8U",
             # 11 檔已下市或已離開
             "COSC": "F83", "SMTC": "S53", "NOLS": "N03", "CMA": "JS8", "OLAM": "O32", "NOBE": "N21",
             "CAPT": "C61U", "JSTG": "J37", "DBSL": "C31", "MGLP": "MC0", "SGPS": "T39"}

# 只保留會用到的欄位
FIELDS = ["RZ", "P", "UP", "PO", "MV", "VO", "UVO", "VA", "NOSH", "UDD", "EPS1FD12", "MTBV", "WC03501",
          "PE", "EPS", "PH", "PL", "DY"]   # DY 只用在幣別反推（股息加總 / UP / DY）

# 因子：代號 → 預期方向
FACTOR_SIGN = {
    "mom20": -1, "mom60": 1,   # mom20 當 20 日反轉用
    "mom12_1": 1, "rev5": 1,
    "vol20": -1, "down60": -1, "beta60": -1, "mdd60": -1,
    "turnover20": -1, "amihud20": -1, "size": -1,
    "dy": 1, "fep": 1, "rev": 1,
    "ep_ttm": 1, "ivol60": -1, "max20": -1,
    "overnight20": -1, "intraday20": 1,   # 依訓練期 IC 的正負號決定
}
if USE_BM:
    FACTOR_SIGN["bm"] = 1
FACTORS = list(FACTOR_SIGN)

# 預處理 / 單因子檢定
MIN_STOCKS = 10
MAD_K = 3
IC_DECAY_WEEKS = (1, 2, 4, 8, 13)
TRADABLE_LOOKBACK = 20; TRADABLE_MIN_DAYS = 5
EP_TTM_LAG = 5          # ep_ttm 用 t−5 的 PE
REG_MIN_FRAC = 0.8      # ivol60 的迴歸至少要有窗口 80% 的有效天數
NEW_FACTORS = ["ep_ttm", "ivol60", "max20", "overnight20", "intraday20"]
