import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
from typing import Optional, Tuple, Dict
import pandas as pd
import re
import threading
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager as fm, rcParams

# 強化 HTTP 取得：帶 User-Agent 以避免站台擋爬（403）
try:
    import requests
except Exception:  # pragma: no cover
    requests = None

# 嘗試載入 yfinance 取得市值；若缺少則在執行時提示安裝
try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # 於執行時檢查並提示使用者

# -------------------------
# Period options
# -------------------------

# 顯示名稱 -> 規格：days 或 ytd, 以及短代號 slug（用於檔名）
PERIOD_SPECS: dict[str, dict] = {
    '1年':     {'days': 365, 'slug': '1Y',  'title': '最近1年'},
    '6個月':   {'days': 182, 'slug': '6M',  'title': '最近6個月'},
    '3個月':   {'days': 90,  'slug': '3M',  'title': '最近3個月'},
    '1個月':   {'days': 30,  'slug': '1M',  'title': '最近1個月'},
    'YTD':     {'ytd': True, 'slug': 'YTD', 'title': '今年以來'},
}

# -------------------------
# Domain logic (from your script)
# -------------------------

def parse_marketcap(mktcap_str: str) -> Optional[float]:
    """
    將市場價值字串（例如 '4.47T', '3.91T', '165.22B'）轉為美元數值（float）。
    支援單位：T（兆）、B（十億）、M（百萬）。
    找不到或格式不符則回傳 None。
    """
    if isinstance(mktcap_str, str):
        m = re.match(r'([\d\.]+)([TBM])', mktcap_str.strip())
        if m:
            value = float(m.group(1))
            unit = m.group(2)
            if unit == 'T':
                return value * 1e12
            elif unit == 'B':
                return value * 1e9
            elif unit == 'M':
                return value * 1e6
    return None


def fetch_constituents_from_web() -> pd.DataFrame:
    """
    從 Wikipedia 抓取 S&P 500 成份股與 GICS 產業。
    https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    回傳含欄位 ['Symbol', 'GICS Sector'] 的 DataFrame。
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = _fetch_html_with_headers(url, referer='https://en.wikipedia.org/')
    tables = _read_html_tables_resilient(html)
    target = None
    for t in tables:
        cols = {str(c).strip() for c in t.columns}
        if {'Symbol', 'GICS Sector'}.issubset(cols):
            target = t
            break
    if target is None:
        raise ValueError('無法從 Wikipedia 抓到包含 Symbol 與 GICS Sector 的表格')
    df = target[['Symbol', 'GICS Sector']].copy()
    # 統一 Symbol 形式
    df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
    return df


def fetch_weights_from_web() -> pd.DataFrame:
    """
    從 Slickcharts 抓取 S&P 500 權重（作為市值權重代理）。
    https://www.slickcharts.com/sp500
    會回傳含欄位 ['Symbol', 'Weight']，其中 Weight 為 0~1 的浮點數。
    """
    url = 'https://www.slickcharts.com/sp500'
    html = _fetch_html_with_headers(url, referer='https://www.slickcharts.com/')
    tables = _read_html_tables_resilient(html)
    target = None
    for t in tables:
        cols = {str(c).strip() for c in t.columns}
        if {'Symbol'}.issubset(cols) and any('Weight' in str(c) for c in t.columns):
            target = t
            break
    if target is None:
        raise ValueError('無法從 Slickcharts 抓到包含 Symbol 與 Weight 的表格')
    # 嘗試尋找權重欄位名稱（可能是 'Weight' 或 'Weight%'）
    weight_col = None
    for c in target.columns:
        if 'Weight' in str(c):
            weight_col = c
            break
    df = target[['Symbol', weight_col]].copy()
    df.columns = ['Symbol', 'Weight']
    # 清理權重：可能含 %
    df['Weight'] = (
        df['Weight']
        .astype(str)
        .str.replace('%', '', regex=False)
        .str.strip()
    )
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    # 若原本是百分比，轉成小數
    # 判斷：若最大值 > 1，視為百分比
    if df['Weight'].max(skipna=True) and df['Weight'].max(skipna=True) > 1:
        df['Weight'] = df['Weight'] / 100.0
    df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
    return df.dropna(subset=['Weight'])


def _fetch_html_with_headers(url: str, referer: Optional[str] = None, retries: int = 2, timeout: int = 15) -> str:
    """
    使用 requests 以常見瀏覽器 User-Agent 取得 HTML，避免 403 Forbidden。
    如無 requests，退回 pandas 直接讀取（可能較容易被擋）。
    """
    if requests is None:
        # 沒有 requests，就讓上層用 pd.read_html(url) 嘗試（風險：403）
        return url
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9,zh-TW;q=0.8',
    }
    if referer:
        headers['Referer'] = referer

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            # 有些站會 403，但加入 header 後通常會 200
            resp.raise_for_status()
            # 嘗試用 UTF-8；若需要可改用 resp.apparent_encoding
            resp.encoding = resp.apparent_encoding or 'utf-8'
            return resp.text
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
            else:
                raise RuntimeError(f'抓取 {url} 失敗：{e}')


def _read_html_tables_resilient(html_or_url: str) -> list[pd.DataFrame]:
    """
    以較穩健的方式將 HTML 解析為表格：先用 lxml；失敗再試 html5lib。
    接受已取回的 HTML 字串或 URL。
    """
    try:
        return pd.read_html(html_or_url, flavor='lxml')
    except Exception:
        try:
            return pd.read_html(html_or_url, flavor='html5lib')
        except Exception:
            # 退回預設（可能再次失敗，但提供最後機會）
            return pd.read_html(html_or_url)


def _period_slug(period_label: str) -> str:
    spec = PERIOD_SPECS.get(period_label, {})
    return str(spec.get('slug', '1Y'))


def _period_title(period_label: str) -> str:
    spec = PERIOD_SPECS.get(period_label, {})
    return str(spec.get('title', period_label))


def _find_return_column(price_df: pd.DataFrame, period_label: str) -> Optional[str]:
    """根據使用者選擇的期間，在價格 Excel 中尋找對應的漲跌幅欄位名。
    會嘗試常見的幾個名稱；若找不到則回傳 None。
    """
    candidates_map: dict[str, list[str]] = {
        '1年': ['1年漲跌幅(%)', '一年漲跌幅(%)', '最近一年(%)', '12個月(%)', '1Y(%)', '1Y'],
        '6個月': ['6個月漲跌幅(%)', '半年漲跌幅(%)', '6M(%)', '6M'],
        '3個月': ['3個月漲跌幅(%)', '三個月漲跌幅(%)', '90天漲跌幅(%)', '3M(%)', '3M', '3月漲跌幅(%)', '3月(%)'],
        '1個月': ['1個月漲跌幅(%)', '一個月漲跌幅(%)', '30天漲跌幅(%)', '1M(%)', '1M', '1月漲跌幅(%)', '1月(%)'],
        'YTD': ['YTD(%)', '年初至今(%)', '今年以來(%)', 'YTD'],
    }
    # 同義詞（報酬率/漲跌幅/回報/return）
    metric_synonyms = ['漲跌幅', '報酬率', '回報', '收益', 'return']

    def _normalize(s: str) -> str:
        # 將全形括號轉半形，移除空白，轉小寫
        trans = str.maketrans({'（': '(', '）': ')', '％': '%'})
        s2 = s.translate(trans)
        s2 = s2.replace(' ', '').lower()
        return s2

    cols = [str(c) for c in price_df.columns]
    # 先精準匹配（大小寫與空白不處理，避免過度偽陽性）
    for cand in candidates_map.get(period_label, []):
        for col in cols:
            if cand == col:
                return col

    # 次級：精準忽略大小寫與空白（normalize 後比對完全相等）
    normalized_candidates = {_normalize(c) for c in candidates_map.get(period_label, [])}
    for col in cols:
        if _normalize(col) in normalized_candidates:
            return col

    # 寬鬆：包含關鍵字（期間 token + metric 同義詞）
    period_tokens_map = {
        '1年': ['1年', '一年', '12個月', '1y'],
        '6個月': ['6個月', '半年', '6m'],
        '3個月': ['3個月', '三個月', '90天', '3m', '3月'],
        '1個月': ['1個月', '一個月', '30天', '1m', '1月'],
        'YTD': ['ytd', '年初至今', '今年以來'],
    }
    ptoks = [t.lower() for t in period_tokens_map.get(period_label, [])]
    for col in cols:
        cn = _normalize(col)
        if any(pt in cn for pt in ptoks) and any(ms in cn for ms in metric_synonyms):
            return col
    return None


def compute_returns_yf(symbols: list[str], period_label: str) -> pd.DataFrame:
    """以 yfinance 計算指定期間的報酬率（百分比）。回傳欄位 ['Symbol', 'Return(%)']。
    僅當 Excel 中沒有相對應欄位時使用。
    """
    if yf is None:
        raise RuntimeError('缺少 yfinance，請先安裝：pip install yfinance')
    if not symbols:
        return pd.DataFrame(columns=['Symbol', 'Return(%)'])
    # 期間解析
    spec = PERIOD_SPECS.get(period_label, PERIOD_SPECS['1年'])
    end = datetime.now()
    if spec.get('ytd'):
        start = datetime(end.year, 1, 1)
    else:
        days = int(spec.get('days', 365))
        start = end - timedelta(days=days)

    # Yahoo 格式映射
    ymap = {sym: _yahoo_symbol(sym) for sym in symbols}
    ylist = list({v for v in ymap.values() if v})
    if not ylist:
        return pd.DataFrame(columns=['Symbol', 'Return(%)'])

    try:
        tickers = yf.Tickers(' '.join(ylist))
        # 使用日期區間抓歷史資料
        hist = tickers.history(start=start, end=end, auto_adjust=True, actions=False, group_by='ticker')
    except Exception:
        # 若批次抓取失敗，回傳空集合（避免整體中斷）
        return pd.DataFrame(columns=['Symbol', 'Return(%)'])

    rows = []
    for orig, ysym in ymap.items():
        try:
            if isinstance(hist.columns, pd.MultiIndex) and ysym in hist.columns.get_level_values(0):
                close = hist[ysym]['Close'].dropna()
            else:
                # 單一標的或非 MultiIndex 時
                close = hist['Close'].dropna()
            if close.empty:
                continue
            first = float(close.iloc[0])
            last = float(close.iloc[-1])
            if first <= 0:
                continue
            ret = (last / first - 1.0) * 100.0  # 以百分比表示
            rows.append({'Symbol': orig, 'Return(%)': ret})
        except Exception:
            continue
    return pd.DataFrame(rows)


def _yahoo_symbol(sym: str) -> str:
    """將部分符號轉換為 Yahoo Finance 接受的格式（如 BRK.B -> BRK-B）。"""
    s = sym.upper().strip()
    return s.replace('.', '-')


def fetch_market_caps_yf(symbols: list[str]) -> pd.DataFrame:
    """
    以 yfinance 取得多檔股票的市值（USD）。
    回傳欄位：['Symbol', 'Market Cap USD']。
    可能因 API/限制導致部分股票無市值，呼叫端須自我處理覆蓋率不足情況。
    """
    if yf is None:
        raise RuntimeError('缺少 yfinance，請先安裝：pip install yfinance')

    # Yahoo 格式轉換與對照
    ymap = {sym: _yahoo_symbol(sym) for sym in symbols}
    ylist = list({v for v in ymap.values() if v})
    if not ylist:
        return pd.DataFrame(columns=['Symbol', 'Market Cap USD'])

    # 批次抓取
    tickers = yf.Tickers(' '.join(ylist))
    out_rows = []
    for orig, ysym in ymap.items():
        tk = tickers.tickers.get(ysym)
        if not tk:
            continue
        cap = None
        # 優先 fast_info，其次 info
        try:
            finfo = getattr(tk, 'fast_info', None)
            if finfo is not None:
                cap = getattr(finfo, 'market_cap', None)
        except Exception:
            cap = None
        if cap is None:
            try:
                info = tk.info  # 可能較慢
                cap = info.get('marketCap') if isinstance(info, dict) else None
            except Exception:
                cap = None
        if cap:
            out_rows.append({'Symbol': orig, 'Market Cap USD': float(cap)})
    return pd.DataFrame(out_rows)


def run_analysis(price_xlsx: Path, constituents_csv: Optional[Path] = None, stockanalysis_html: Optional[Path] = None,
                 use_online_data: bool = False, period_label: str = '1年') -> Tuple[pd.Series, Dict, pd.DataFrame]:
    """
    執行產業市值加權一年漲幅與貢獻度分析。
    回傳：
      - sector_returns: pd.Series（index=GICS Sector, values=百分比）
      - sector_contrib: dict（每個產業的主要公司貢獻）
    """
    # 1) 讀取資料
    price_df = pd.read_excel(price_xlsx)

    if use_online_data:
        # 線上取得成份股/產業，並嘗試以 yfinance 取得市值；若覆蓋率不足再回退改用 Slickcharts 權重
        constituents = fetch_constituents_from_web()
        # 市值
        caps_df = fetch_market_caps_yf(constituents['Symbol'].tolist())
        market = caps_df if not caps_df.empty else None
        weights = None
        coverage = 0.0
        if market is not None:
            coverage = market['Symbol'].nunique() / max(1, constituents['Symbol'].nunique())
        if market is None or coverage < 0.6:
            # 覆蓋率不足，改用權重
            weights = fetch_weights_from_web()
            market = None
    else:
        if constituents_csv is None or stockanalysis_html is None:
            raise ValueError('未提供成份股 CSV 或市值 HTML，且未開啟線上模式。')
        constituents = pd.read_csv(constituents_csv)
        sp500_tables = pd.read_html(str(stockanalysis_html))
        market_table = None
        for tbl in sp500_tables:
            if 'Symbol' in tbl.columns and 'Market Cap' in tbl.columns:
                market_table = tbl
                break
        if market_table is None:
            raise ValueError('未找到包含市值資訊的表格（需含欄位 Symbol 與 Market Cap）。')
        market = market_table[['Symbol', 'Market Cap']].copy()
        market['Market Cap USD'] = market['Market Cap'].apply(parse_marketcap)
        market = market.dropna(subset=['Market Cap USD']).drop(columns=['Market Cap'])

    # 2) 整併
    if 'Symbol' not in price_df.columns:
        raise ValueError("價格檔缺少欄位 'Symbol'")
    # 依所選期間取得報酬率欄位；若無，且為線上模式則改以 yfinance 計算
    ret_col = _find_return_column(price_df, period_label)
    if ret_col is None:
        if use_online_data:
            # 以 yfinance 計算
            try:
                ret_df_yf = compute_returns_yf(constituents['Symbol'].tolist(), period_label)
            except Exception as e:
                raise RuntimeError(f'計算 {period_label} 報酬率失敗：{e}')
            if ret_df_yf.empty:
                raise RuntimeError(f'找不到 {period_label} 報酬率欄位，且無法線上計算。請確認 yfinance 可用或提供含該期間欄位的 Excel。')
            ret_df = ret_df_yf.rename(columns={'Return(%)': '選定期間漲跌幅(%)'})
        else:
            raise ValueError(f"價格檔缺少所選期間的欄位（{period_label}），請在 Excel 中加入或改用線上模式。")
    else:
        # 從 Excel 取得
        ret_df = price_df[['Symbol', ret_col]].copy()
        ret_df = ret_df.rename(columns={ret_col: '選定期間漲跌幅(%)'})
    if not {'Symbol', 'GICS Sector'}.issubset(constituents.columns):
        raise ValueError("成份股檔缺少欄位 'Symbol' 或 'GICS Sector'")

    merged = price_df[['Symbol']].merge(constituents[['Symbol', 'GICS Sector']], on='Symbol', how='left')
    merged = merged.merge(ret_df[['Symbol', '選定期間漲跌幅(%)']], on='Symbol', how='left')
    # 若有市值，走市值加權；否則使用權重
    if market is not None:
        merged = merged.merge(market[['Symbol', 'Market Cap USD']], on='Symbol', how='left')

    merged['選定期間漲跌幅(%)'] = pd.to_numeric(merged['選定期間漲跌幅(%)'], errors='coerce')
    merged['return'] = merged['選定期間漲跌幅(%)'].fillna(0) / 100

    if market is not None:
        # 有市值：以市值為基礎計算
        merged = merged.dropna(subset=['Market Cap USD'])
        sector_marketcap = merged.groupby('GICS Sector')['Market Cap USD'].sum()
        merged['weight'] = merged['Market Cap USD'] / merged['GICS Sector'].map(sector_marketcap)
        merged['contribution'] = merged['weight'] * merged['return']
        sector_returns = (
            merged.groupby('GICS Sector')['contribution']
            .sum()
            .sort_values(ascending=False) * 100
        )
    else:
        # 無市值：改用 Slickcharts 權重或線上權重（以指數權重作為市值代理），
        # 並在產業內部重新正規化（保持產業加權為 1）
        weights = locals().get('weights')  # 取自上面 use_online_data 分支
        merged = merged.merge(weights[['Symbol', 'Weight']], on='Symbol', how='left')
        merged = merged.dropna(subset=['Weight'])
        # 產業內正規化
        sector_weight_sum = merged.groupby('GICS Sector')['Weight'].transform('sum')
        merged['weight'] = merged['Weight'] / sector_weight_sum
        merged['contribution'] = merged['weight'] * merged['return']
        sector_returns = (
            merged.groupby('GICS Sector')['contribution']
            .sum()
            .sort_values(ascending=False) * 100
        )

    # 4) 公司對產業貢獻度
    sector_contrib: dict[str, dict] = {}
    for sector, grp in merged.groupby('GICS Sector'):
        total_abs = grp['contribution'].abs().sum()
        if total_abs == 0:
            # 避免除以 0
            continue
        grp = grp.assign(
            share_abs=grp['contribution'].abs() / total_abs,
            sign=grp['contribution'].apply(lambda x: 'pos' if x >= 0 else 'neg')
        )
        top = grp.sort_values('share_abs', ascending=False)
        top_entries = top.iloc[:5][['Symbol', 'share_abs', 'sign']]
        others_share = 1 - top_entries['share_abs'].sum()
        sector_contrib[sector] = {
            'top_companies': list(top_entries['Symbol']),
            'top_shares': list((top_entries['share_abs'] * 100).round(2)),
            'signs': list(top_entries['sign']),
            'others_share': round(float(others_share * 100), 2),
        }

    return sector_returns, sector_contrib, merged


def _setup_cjk_font() -> Optional[str]:
    """設定可顯示中文的字型並關閉 unicode 負號，避免圖表亂碼。"""
    try:
        available = {f.name for f in fm.fontManager.ttflist}
        candidates = [
            'Microsoft JhengHei', 'Microsoft YaHei', 'SimHei',
            'PingFang TC', 'PingFang SC',
            'Noto Sans CJK TC', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
            'Arial Unicode MS', 'PMingLiU', 'MingLiU',
        ]
        chosen = next((c for c in candidates if c in available), None)
        if chosen:
            rcParams['font.sans-serif'] = [chosen] + list(rcParams.get('font.sans-serif', []))
        rcParams['axes.unicode_minus'] = False
        return chosen
    except Exception:
        rcParams['axes.unicode_minus'] = False
        return None


def _safe_filename(name: str) -> str:
    invalid = '<>:"/\\|?*'
    safe = ''.join('_' if ch in invalid else ch for ch in str(name))
    safe = '_'.join(s for s in safe.strip().split())
    return safe or 'untitled'


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    p = path
    i = 1
    while True:
        cand = p.with_name(f"{p.stem} ({i}){p.suffix}")
        if not cand.exists():
            return cand
        i += 1


def _append_timestamp(path: Path, dt: Optional[datetime] = None) -> Path:
    """在檔名後面加上分析時間，例如 sector_returns_20251021_1435.png。"""
    t = dt or datetime.now()
    suffix = t.strftime('%Y%m%d_%H%M')
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")

def _timestamp_str(dt: Optional[datetime] = None) -> str:
    """回傳 YYYYMMDD_HHMM 的時間戳字串。"""
    t = dt or datetime.now()
    return t.strftime('%Y%m%d_%H%M')

def _prepend_datetime_to_name(base_name: str, dt: Optional[datetime] = None) -> str:
    """在檔名最前面加上日期時間，如 20251125_1430.BaseName（自動清理不合法字元）。"""
    ts = _timestamp_str(dt)
    return f"{ts}.{_safe_filename(base_name)}"


def plot_sector_returns_bar(sector_returns: pd.Series, out_img: Path, title: str = 'S&P 500 各產業加權漲跌幅', period_title: Optional[str] = None) -> None:
    """輸出整體產業加權漲跌幅長條圖。可加上期間說明。"""
    _setup_cjk_font()
    sr = sector_returns.sort_values(ascending=True)
    colors = ['#ef4444' if v < 0 else '#10b981' for v in sr.values]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.barh(sr.index, sr.values, color=colors, edgecolor='#1f2937')
    ax.set_xlabel('報酬率 (%)')
    ttl = title if not period_title else f"{title}（{period_title}）"
    ax.set_title(ttl)
    ax.grid(axis='x', alpha=0.25)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
    for y, v in enumerate(sr.values):
        ax.text(v + (0.6 if v >= 0 else -0.6), y, f'{v:.1f}%', va='center', ha='left' if v >= 0 else 'right', fontsize=9)
    fig.tight_layout()
    out_img.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_img, bbox_inches='tight')
    plt.close(fig)


def plot_sector_breakdowns(merged: pd.DataFrame, out_dir: Path, top_n: Optional[int] = None, dt: Optional[datetime] = None,
                           period_slug: Optional[str] = None, period_title: Optional[str] = None) -> Dict[str, Path]:
    """輸出每個產業的成分股一年漲跌幅圖，回傳 {產業: 圖檔路徑}。
    檔名規則：YYYYMMDD_HHMM.<產業名>.png（Windows 不合法字元自動清理）。
    """
    _setup_cjk_font()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}
    ts = _timestamp_str(dt)
    for sector, grp in merged.groupby('GICS Sector'):
        df = grp.copy()
        df['Return(%)'] = df['return'] * 100
        if top_n is not None and top_n > 0:
            df = df.assign(abs_contrib=df['contribution'].abs()).sort_values('abs_contrib', ascending=False).head(top_n)
        df = df.dropna(subset=['Return(%)'])
        if df.empty:
            continue
        df = df.sort_values('Return(%)')
        symbols = df['Symbol'].astype(str).tolist()
        values = df['Return(%)'].astype(float).tolist()
        colors = ['#ef4444' if v < 0 else '#10b981' for v in values]
        height = max(4.5, min(0.3 * len(df) + 1.5, 30))
        fig, ax = plt.subplots(figsize=(12, height), dpi=120)
        ax.barh(symbols, values, color=colors, edgecolor='#1f2937', linewidth=0.6)
        ax.set_xlabel('報酬率 (%)')
        ttl = f'{sector} 成分股漲跌幅'
        if period_title:
            ttl = f'{ttl}（{period_title}）'
        ax.set_title(ttl)
        ax.grid(axis='x', alpha=0.25)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
        if len(df) <= 30:
            for y, v in enumerate(values):
                ax.text(v + (0.5 if v >= 0 else -0.5), y, f'{v:.1f}%', va='center', ha='left' if v >= 0 else 'right', fontsize=8)
        fig.tight_layout()
        slug = period_slug or _period_slug('1年')
        fname = f"{ts}.{slug}.{_safe_filename(sector)}.png"
        fpath = _unique_path(out_dir / fname)
        fig.savefig(fpath, bbox_inches='tight')
        plt.close(fig)
        results[str(sector)] = fpath
    return results


# -------------------------
# Simple Tk GUI
# -------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('S&P 500 產業一年漲跌幅（市值加權）- GUI')
        self.geometry('880x640')
        self.minsize(760, 560)

        # Paths
        self.var_price = tk.StringVar()
        self.var_const = tk.StringVar()
        self.var_html = tk.StringVar()
        self.var_online = tk.BooleanVar(value=True)
        self.var_period = tk.StringVar(value='1年')

        # Layout
        self._build_inputs()
        self._build_actions()
        self._build_output()

        # Try to prefill common folder
        self._prefill_initial_dir()

    def _build_inputs(self):
        frm = ttk.LabelFrame(self, text='輸入檔案')
        frm.pack(fill='x', padx=12, pady=10)

        # 線上模式選項（只需價格 Excel）
        chk = ttk.Checkbutton(frm, text='使用線上成份/產業與市值（只需提供價格變動 Excel）', variable=self.var_online,
                      command=self._toggle_online)
        chk.pack(anchor='w', padx=10, pady=(8, 2))

        # 期間選擇
        row_period = ttk.Frame(frm)
        row_period.pack(fill='x', padx=10, pady=4)
        ttk.Label(row_period, text='分析期間', width=28).pack(side='left')
        self.cbo_period = ttk.Combobox(row_period, textvariable=self.var_period, state='readonly',
                                       values=list(PERIOD_SPECS.keys()))
        self.cbo_period.pack(side='left', fill='x', expand=True)

        # Price Excel
        self._row_file_picker(frm, '價格變動 Excel', self.var_price,
                      [('Excel', '*.xlsx;*.xls')])
        # Constituents CSV
        self.row_const = self._row_file_picker(frm, '成份股與產業 CSV', self.var_const,
                      [('CSV', '*.csv')])
        # StockAnalysis HTML
        self.row_html = self._row_file_picker(frm, 'StockAnalysis HTML（含市值表格）', self.var_html,
                      [('HTML', '*.html;*.htm')])
        # 初始切換一次
        self.after(100, self._toggle_online)

    def _row_file_picker(self, parent, label, var, filetypes):
        row = ttk.Frame(parent)
        row.pack(fill='x', padx=10, pady=8)
        ttk.Label(row, text=label, width=28).pack(side='left')
        ent = ttk.Entry(row, textvariable=var)
        ent.pack(side='left', fill='x', expand=True)
        def browse():
            path = filedialog.askopenfilename(title=label, filetypes=filetypes)
            if path:
                var.set(path)
        btn = ttk.Button(row, text='瀏覽…', command=browse)
        btn.pack(side='left', padx=6)
        return {'entry': ent, 'button': btn}

    def _toggle_online(self):
        # 切換是否使用線上資料；若是，則停用 CSV/HTML 選擇列
        online = self.var_online.get()
        state = 'disabled' if online else 'normal'
        for row in [getattr(self, 'row_const', None), getattr(self, 'row_html', None)]:
            if not row:
                continue
            try:
                row['entry'].configure(state=state)
                row['button'].configure(state=state)
            except Exception:
                pass

    def _build_actions(self):
        frm = ttk.Frame(self)
        frm.pack(fill='x', padx=12, pady=6)
        ttk.Button(frm, text='執行分析', command=self.on_run).pack(side='left')
        ttk.Button(frm, text='儲存結果到 CSV', command=self.on_save_csv).pack(side='left', padx=8)
        ttk.Button(frm, text='輸出主圖', command=self.on_export_main_chart).pack(side='left', padx=8)
        ttk.Button(frm, text='輸出各產業附圖', command=self.on_export_sector_charts).pack(side='left', padx=8)
        ttk.Button(frm, text='清空輸出', command=lambda: self.txt_output.delete('1.0', 'end')).pack(side='left')

    def _build_output(self):
        lbl = ttk.Label(self, text='輸出：')
        lbl.pack(anchor='w', padx=14)
        self.txt_output = ScrolledText(self, wrap='word', height=22)
        self.txt_output.pack(fill='both', expand=True, padx=12, pady=6)
        self.txt_output.configure(font=('Consolas', 10))

    def _prefill_initial_dir(self):
        # 若常用資料夾存在，可預填，否則留空讓使用者瀏覽
        # 你可以依需求修改以下路徑預設
        common_dir = Path.home() / 'Documents' / 'Python' / '股票分析'
        if common_dir.exists():
            # 提示路徑，未自動帶入檔名以避免錯誤
            self.txt_output.insert('end', f"建議從這個資料夾挑選檔案：{common_dir}\n\n")

    def _validate_paths(self):
        p = Path(self.var_price.get()) if self.var_price.get() else None
        online = self.var_online.get()
        if not (p and p.exists()):
            raise FileNotFoundError('請提供價格變動 Excel')
        if online:
            return p, None, None
        c = Path(self.var_const.get()) if self.var_const.get() else None
        h = Path(self.var_html.get()) if self.var_html.get() else None
        missing = []
        if not (c and c.exists()): missing.append('成份股與產業 CSV')
        if not (h and h.exists()): missing.append('StockAnalysis HTML')
        if missing:
            raise FileNotFoundError('下列檔案不存在或未選擇：' + ', '.join(missing))
        return p, c, h

    def on_run(self):
        try:
            p, c, h = self._validate_paths()
            sector_returns, sector_contrib, merged = run_analysis(
                p, c, h,
                use_online_data=self.var_online.get(),
                period_label=self.var_period.get()
            )
            self._last_merged = merged
            self._render_results(sector_returns, sector_contrib, self.var_period.get())
            messagebox.showinfo('完成', '分析已完成')
        except Exception as e:
            messagebox.showerror('錯誤', str(e))

    def _render_results(self, sector_returns: pd.Series, sector_contrib: dict, period_label: Optional[str] = None):
        self.txt_output.delete('1.0', 'end')
        ptitle = _period_title(period_label or '1年')
        self.txt_output.insert('end', f'各產業市值加權漲幅（%） - {ptitle}:\n')
        # 對齊輸出
        maxlen = max((len(str(idx)) for idx in sector_returns.index), default=10)
        for sector, val in sector_returns.items():
            self.txt_output.insert('end', f"  {sector:<{maxlen}} : {val:.2f}%\n")
        self.txt_output.insert('end', '\n產業主要公司貢獻（占比%）：\n')
        for sector, info in sector_contrib.items():
            self.txt_output.insert('end', f"{sector}:\n")
            for sym, share, sgn in zip(info['top_companies'], info['top_shares'], info['signs']):
                direction = '正貢獻' if sgn == 'pos' else '負貢獻'
                self.txt_output.insert('end', f"  {sym}: {share}% ({direction})\n")
            self.txt_output.insert('end', f"  Others: {info['others_share']}%\n")
        self.txt_output.see('end')

    def on_save_csv(self):
        # 允許將 sector_returns 存為 CSV
        try:
            p, c, h = self._validate_paths()
            sector_returns, sector_contrib, _ = run_analysis(
                p, c, h,
                use_online_data=self.var_online.get(),
                period_label=self.var_period.get()
            )
            out_path = filedialog.asksaveasfilename(
                title='儲存產業漲跌幅 CSV',
                defaultextension='.csv',
                filetypes=[('CSV', '*.csv')]
            )
            if not out_path:
                return
            slug = _period_slug(self.var_period.get())
            sr = sector_returns.rename(f'Return(%)_{slug}').to_frame()
            sr.to_csv(out_path, encoding='utf-8-sig')
            messagebox.showinfo('已儲存', f'已儲存至 {out_path}')
        except Exception as e:
            messagebox.showerror('錯誤', str(e))

    def on_export_main_chart(self):
        try:
            p, c, h = self._validate_paths()
            sector_returns, sector_contrib, merged = run_analysis(
                p, c, h,
                use_online_data=self.var_online.get(),
                period_label=self.var_period.get()
            )
            ts = _timestamp_str()
            slug = _period_slug(self.var_period.get())
            out_path = filedialog.asksaveasfilename(
                title='儲存主圖（整體產業漲跌幅）PNG',
                defaultextension='.png',
                filetypes=[('PNG Image', '*.png')],
                initialfile=f'{ts}.{slug}.sector_returns.png'
            )
            if not out_path:
                return
            out_img = _unique_path(Path(out_path))
            plot_sector_returns_bar(sector_returns, out_img, period_title=_period_title(self.var_period.get()))
            self.txt_output.insert('end', f"\n主圖已輸出：{out_img}\n")
            self.txt_output.see('end')
            messagebox.showinfo('完成', f'主圖已輸出到\n{out_img}')
        except Exception as e:
            messagebox.showerror('錯誤', str(e))

    def on_export_sector_charts(self):
        try:
            p, c, h = self._validate_paths()
            # 若已有先前分析結果且仍在相同輸入下，直接使用；否則重新跑一次
            sector_returns, sector_contrib, merged = run_analysis(
                p, c, h,
                use_online_data=self.var_online.get(),
                period_label=self.var_period.get()
            )
            out_dir = filedialog.askdirectory(title='選擇各產業附圖輸出資料夾')
            if not out_dir:
                return
            out_dir_path = Path(out_dir)
            # 使用同一個時間戳，方便之後做時間軸比對
            dt_now = datetime.now()
            results = plot_sector_breakdowns(
                merged, out_dir_path, dt=dt_now,
                period_slug=_period_slug(self.var_period.get()),
                period_title=_period_title(self.var_period.get())
            )
            if results:
                self.txt_output.insert('end', f"\n已輸出各產業附圖至：{out_dir_path}\n")
                for sec, pth in results.items():
                    self.txt_output.insert('end', f"  - {sec}: {pth}\n")
                self.txt_output.see('end')
                messagebox.showinfo('完成', f'各產業附圖已輸出到\n{out_dir_path}')
            else:
                messagebox.showwarning('提醒', '未產生任何產業附圖，可能因資料不足。')
        except Exception as e:
            messagebox.showerror('錯誤', str(e))


if __name__ == '__main__':
    app = App()
    app.mainloop()
