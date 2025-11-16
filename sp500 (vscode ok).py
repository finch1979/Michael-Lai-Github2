#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S&P 500 價格變化報告（匯出 Excel）

功能:
1. 下載 S&P500 成分股。
2. 抓取股價歷史資料。
3. 計算 1日, 1週, 1月, 3月, 半年, 1年, 2年 漲跌幅。
4. 儲存為 Excel 報告（若缺少 openpyxl 則改存 CSV）。
"""

from __future__ import annotations
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_sp500_symbols() -> list[str]:
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(url)
    syms = (
        df["Symbol"].dropna().astype(str).str.strip().unique().tolist()
    )
    # yfinance 使用 - 代替 . 例如 BRK.B -> BRK-B, BF.B -> BF-B
    syms = [s.replace(".", "-") for s in syms]
    return syms


def calc_price_changes(symbol: str, start_date, end_date, horizon_days: dict[str, int]) -> dict | None:
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty or "Close" not in df.columns:
            return None
        closes = df["Close"].dropna()
        if closes.empty:
            return None
        row: dict = {"Symbol": symbol}
        for label, days in horizon_days.items():
            if len(closes) >= days + 1:
                price_now = float(closes.iloc[-1])
                price_before = float(closes.iloc[-(days + 1)])
                if price_before != 0:
                    change = (price_now - price_before) / price_before * 100.0
                    row[f"{label}漲跌幅(%)"] = round(change, 2)
                else:
                    row[f"{label}漲跌幅(%)"] = None
            else:
                row[f"{label}漲跌幅(%)"] = None
        return row
    except Exception as e:
        print(f"⚠️ {symbol} 下載失敗: {e}")
        return None


def main():
    # 儲存位置
    output_path = Path.home() / "Documents" / "MarketReports"
    output_path.mkdir(parents=True, exist_ok=True)

    # 計算時間範圍（交易日近似）
    horizon_days = {
        "1日": 1,
        "1週": 5,
        "1月": 22,
        "3月": 66,
        "半年": 130,
        "1年": 260,
        "2年": 520,
    }

    start_date = datetime.today().date() - timedelta(days=max(horizon_days.values()) + 30)
    end_date = datetime.today().date()

    # 下載清單
    symbols = fetch_sp500_symbols()
    print(f"共 {len(symbols)} 檔，開始抓取...\n")

    result_rows: list[dict] = []
    for i, sym in enumerate(symbols, 1):
        row = calc_price_changes(sym, start_date, end_date, horizon_days)
        if row:
            result_rows.append(row)
        if i % 25 == 0:
            print(f"進度 {i}/{len(symbols)}...")
        # 稍作停頓，避免被遠端限流
        time.sleep(0.05)

    if not result_rows:
        print("⚠️ 沒有成功的資料可輸出。")
        return

    result_df = pd.DataFrame(result_rows)
    date_tag = datetime.today().strftime("%Y%m%d")
    excel_path = output_path / f"sp500_price_changes_{date_tag}.xlsx"

    # 儲存 Excel（若無 openpyxl 則改存 CSV）
    try:
        result_df.to_excel(excel_path, index=False)
        print(f"✅ 已儲存報告: {excel_path}")
    except Exception as e:
        csv_path = output_path / f"sp500_price_changes_{date_tag}.csv"
        result_df.to_csv(csv_path, index=False)
        print(f"⚠️ Excel 輸出失敗（{e}），已改存 CSV: {csv_path}")


if __name__ == "__main__":
    main()
