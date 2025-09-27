# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 01:26:16 2025

@author: Michael Lai
"""
from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Analyze selected data-center vendor tickers with yfinance.
- Uses a hard-coded table (公司, 股票代碼)，自動拆分多重代碼（如 "TT / IR"、或含逗號）
- 計算 一天、 一週、 一個月、 一季、 半年、 一年 漲跌幅（以營業日近似：1,5,22,66,130,260）
- 以 yfinance 下載約 3 年日線（自動使用調整收盤價）
- 輸出 Excel 與 CSV 到 ~/Documents/MarketReports 與 /mnt/data

注意：
- Schneider Electric 在 yfinance 的常用代碼為 "SU.PA"（巴黎交易所）；若使用 "SU" 會抓到 Suncor Energy。
- Daikin OTC: "DAIKY"。
"""
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ------------------------------
# 1) 原始清單（依你提供的表）
# ------------------------------
RAW_ROWS = [
    ("Dell", "DELL"),
    ("HPE", "HPE"),
    ("Super Micro", "SMCI"),
    ("Arista Networks", "ANET"),
    ("Cisco", "CSCO"),
    ("NVIDIA", "NVDA"),
    ("SPX Technologies", "SPXC"),
    ("Johnson Controls", "JCI"),
    ("Trane / Ingersoll Rand", "TT / IR"),
    ("Carrier", "CARR"),
    ("Daikin", "DAIKY"),
    ("Vertiv", "VRT"),
    ("Schneider", "SU"),   # 會在修正表中改為 SU.PA
    ("Eaton", "ETN"),
]

# ------------------------------
# 2) 代碼修正（若需要跨交易所常見寫法）
# ------------------------------
SYMBOL_FIXES: Dict[str, str] = {
    "SU": "SU.PA",   # Schneider Electric on Euronext Paris
    # 其他需要修正的可以放這裡
}

# ------------------------------
# 3) 參數設定
# ------------------------------
HORIZONS = {
    "1日": 1,
    "1週": 5,
    "1月": 22,
    "1季": 66,   # 約 3 個月（1 季）
    "半年": 130,
    "1年": 260,
}

OUTPUT_DIRS = [
    Path.home() / "Documents" / "MarketReports",
    Path("/mnt/data"),
]

# ------------------------------
# 4) 工具函式
# ------------------------------
SEP_CHARS = ["/", ",", "、", "|", " "]


def explode_symbols(name: str, raw_codes: str) -> List[Dict[str, str]]:
    """將一列（公司, 代碼字串）拆成多個乾淨的單一代碼列。
    返回 [{'公司': name, 'Symbol': code}, ...]
    """
    s = raw_codes.strip()
    for sep in ["/", ",", "、", "|"]:
        s = s.replace(sep, " ")
    parts = [p.strip() for p in s.split() if p.strip()]
    rows = []
    for p in parts:
        fixed = SYMBOL_FIXES.get(p, p)
        rows.append({"公司": name, "Symbol": fixed})
    return rows


def pct_change_by_bdays(closes: pd.Series, bdays: int) -> Optional[float]:
    """從最後一筆收盤價回推 b 個營業日的變化百分比。"""
    if closes.empty or not isinstance(closes.index, pd.DatetimeIndex):
        return None
    s = closes.sort_index().dropna()
    if s.size < bdays + 1:
        return None
    last_date = s.index[-1]
    ref_idx = pd.bdate_range(end=last_date, periods=bdays + 1)[0]
    ref = s.loc[:ref_idx]
    if ref.empty:
        return None
    price_now = float(s.iloc[-1])
    price_before = float(ref.iloc[-1])
    if price_before == 0 or np.isnan(price_before):
        return None
    return round((price_now - price_before) / price_before * 100.0, 2)

# ------------------------------
# 5) 主流程
# ------------------------------

def main():
    # 準備展開後的代碼表
    rows: List[Dict[str, str]] = []
    for name, raw_code in RAW_ROWS:
        rows.extend(explode_symbols(name, raw_code))
    base_df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

    results = []
    for _, r in base_df.iterrows():
        comp, sym = r["公司"], r["Symbol"]
        try:
            hist = yf.download(sym, period="3y", interval="1d", auto_adjust=True, progress=False)
            if hist.empty or "Close" not in hist.columns:
                print(f"⚠️ 無資料：{sym}")
                continue
            closes = hist["Close"].dropna()
            row = {
                "公司": comp,
                "Symbol": sym,
                "最新收盤日": closes.index[-1].strftime("%Y-%m-%d"),
                "最新收盤價": round(float(closes.iloc[-1]), 4),
            }
            for label, days in HORIZONS.items():
                row[f"{label}漲跌幅(%)"] = pct_change_by_bdays(closes, days)
            results.append(row)
        except Exception as e:
            print(f"⚠️ 下載失敗 {sym}: {e}")

    if not results:
        print("❌ 沒有成功下載任何標的，請檢查代碼與網路連線。")
        return

    out_df = pd.DataFrame(results)
    # 排序：預設依 1年漲跌幅 由高到低
    if "1年漲跌幅(%)" in out_df.columns:
        out_df = out_df.sort_values("1年漲跌幅(%)", ascending=False)

    # 輸出
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"selected_tickers_price_changes_{ts}"

    for d in OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        xls = d / f"{fname}.xlsx"
        csv = d / f"{fname}.csv"
        out_df.to_excel(xls, index=False)
        out_df.to_csv(csv, index=False, encoding="utf-8-sig")
        print(f"✅ 已輸出：{xls}")
        print(f"✅ 已輸出：{csv}")

    # 預覽列印（一年漲幅前 12 名）
    cols = ["公司", "Symbol", "1年漲跌幅(%)", "最新收盤價", "最新收盤日"]
    print("\n📊 一年漲幅前 12 名預覽：")
    with pd.option_context("display.max_rows", 20, "display.max_columns", 20):
        print(out_df[[c for c in cols if c in out_df.columns]].head(12))


if __name__ == "__main__":
    main()
