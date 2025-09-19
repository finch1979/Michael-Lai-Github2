# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 01:24:44 2025

@author: Michael Lai
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 19:46:48 2025

@author: Michael Lai
"""

import subprocess
import sys
import importlib

# 自動檢查套件，沒有就安裝
def install_if_missing(package_name):
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} 已安裝")
    except ImportError:
        print(f"⚠️ {package_name} 未安裝，正在安裝...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# 確保安裝必要套件
def check_required_packages():
    required_packages = ['yfinance', 'pandas', 'openpyxl']
    for package in required_packages:
        install_if_missing(package)

# 開始檢查
check_required_packages()

# 從這裡開始原始程式
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 設定儲存路徑
documents_path = Path.home() / "Documents" / "MarketReports"
documents_path.mkdir(parents=True, exist_ok=True)

print(f"當前儲存路徑: {documents_path}")

# 設定標的
symbols = {
    "S&P 500": "^GSPC",
    "台灣 0050": "0050.TW",
    "富邦美債20年": "00696B.TW",
    "國泰半導體": "00830.TW",
    "台積電": "2330.TW"
}

# 計算各時間點漲跌幅
def get_price_change(data, days_ago):
    closes = data['Close'].dropna()
    if len(closes) < days_ago + 1:
        return None, None, None
    close_today = float(closes.iloc[-1])
    close_past = float(closes.iloc[-(days_ago + 1)])
    change = (close_today - close_past) / close_past * 100
    return close_past, close_today, change

# 計算所需天數的開始日期
today = datetime.now().date()
max_days_needed = 1300
start_date = today - timedelta(days=max_days_needed + 10)

results = []

# 下載資料並計算
dataframes = {}
for name, symbol in symbols.items():
    df = yf.download(symbol, start=start_date, end=today + timedelta(days=1))
    dataframes[name] = df

    result = {"指數": name}

    for label, days in [
        ("今日", 1),
        ("一週", 5),
        ("半年", 130),
        ("一年", 260),
        ("兩年", 520),
        ("五年", 1300)
    ]:
        price_start, price_end, change = get_price_change(df, days)
        if price_start is not None:
            result[f"{label}前收盤價"] = f"{price_start:.2f}"
            result[f"{label}漲跌幅"] = f"{change:+.2f}%"
        else:
            result[f"{label}前收盤價"] = "N/A"
            result[f"{label}漲跌幅"] = "N/A"

    results.append(result)

# 整理結果 DataFrame
result_df = pd.DataFrame(results)

# 定義報表名稱
report_filename = documents_path / f"market_report_{today}-2.xlsx"

# 儲存為 Excel
result_df.to_excel(report_filename, index=False)

print(f"✅ 報表已成功儲存為 {report_filename}")