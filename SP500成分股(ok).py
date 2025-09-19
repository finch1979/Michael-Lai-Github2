# 儲存為 DataFrame 並匯出
result_df = pd.DataFrame(result_data)
excel_path = output_path / f"sp500_price_changes_{datetime.today().strftime('%Y%m%d_%H%M%S')}.xlsx"
result_df.to_excel(excel_path, index=False)

print(f"✅ 已儲存報告: {excel_path}")excel_path = output_path / f"sp500_price_changes_{datetime.today().strftime('%Y%m%d')}.xlsx"excel_path = output_path / f"sp500_price_changes_{datetime.today().strftime('%Y%m%d')}.xlsx"excel_path = output_path / f"sp500_price_changes_{datetime.today().strftime('%Y%m%d')}.xlsx"excel_path = output_path / f"sp500_price_changes_{datetime.today().strftime('%Y%m%d')}.xlsx"excel_path = output_path / f"sp500_price_changes_{datetime.today().strftime('%Y%m%d')}.xlsx"# ...existing code...

# 儲存為 DataFrame 並匯出
result_df = pd.DataFrame(result_data)
excel_path = output_path / f"sp500_price_changes_{datetime.today().strftime('%Y%m%d_%H%M%S')}.xlsx"
result_df.to_excel(excel_path, index=False)

print(f"✅ 已儲存報告: {excel_path}")
# ...existing code...# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 02:57:18 2025

@author: Michael Lai
"""

# -*- coding: utf-8 -*-
"""
Created on 2025-09-19
Author: Michael Lai

功能:
1. 下載 S&P500 成分股。
2. 抓取股價歷史資料。
3. 計算 1日, 1週, 1月, 3月, 半年, 1年, 2年 漲跌幅。
4. 輸出為 Excel 報告。
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

# 儲存位置
output_path = Path.home() / "Documents" / "MarketReports"
output_path.mkdir(parents=True, exist_ok=True)

# 下載 S&P500 成分股清單
sp500_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
sp500_df = pd.read_csv(sp500_url)

# 建立 symbol list
symbols = sp500_df['Symbol'].dropna().unique().tolist()

# 計算時間範圍
horizon_days = {
    '1日': 1,
    '1週': 5,
    '1月': 22,
    '3月': 66,
    '半年': 130,
    '1年': 260,
    '2年': 520
}

start_date = datetime.today().date() - timedelta(days=max(horizon_days.values()) + 30)
end_date = datetime.today().date()

# 儲存所有結果
result_data = []

for symbol in symbols:
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty or 'Close' not in df.columns:
            continue

        closes = df['Close'].dropna()
        row = {'Symbol': symbol}

        for label, days in horizon_days.items():
            if len(closes) >= days + 1:
                price_now = closes.iloc[-1]
                price_before = closes.iloc[-(days + 1)]
                change = (price_now - price_before) / price_before * 100
                row[f'{label}漲跌幅(%)'] = round(change, 2)
            else:
                row[f'{label}漲跌幅(%)'] = None

        result_data.append(row)

    except Exception as e:
        print(f"⚠️ {symbol} failed: {e}")

# 儲存為 DataFrame 並匯出
result_df = pd.DataFrame(result_data)
excel_path = output_path / f"sp500_price_changes_{datetime.today().strftime('%Y%m%d')}.xlsx"
result_df.to_excel(excel_path, index=False)

print(f"✅ 已儲存報告: {excel_path}")
