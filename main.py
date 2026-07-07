import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# パス定義
ROOT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = ROOT_DIR / "docs"

# 複数アセット定義
ASSET_CONFIG = {
    "SI": {"name": "銀先物 (SI)", "ticker": "SI=F", "multiplier": 5000, "filename": "index.html"},
    "NG": {"name": "天然ガス先物 (NG)", "ticker": "NG=F", "multiplier": 10000, "filename": "ng.html"},
    "ZS": {"name": "大豆先物 (ZS)", "ticker": "ZS=F", "multiplier": 5000, "filename": "zs.html"}
}

def parse_strike(val):
    """520-0 や 1.250 などの形式を正確にfloatへ変換"""
    s = str(val).split('-')[0].replace(',', '').strip()
    try: return float(s)
    except: return 0.0

def clean_data_string(val):
    """N/Aや表記揺れをクリーンなfloatへ変換"""
    s = str(val).split('-')[0].replace('%', '').replace(',', '').replace('N/A', '0').replace('nan', '0').strip()
    try: return float(s)
    except: return 0.0

def load_barchart_csv(prefix):
    sb_files = sorted(glob.glob(f"{prefix}*side-by-side*.csv"))
    gk_files = sorted(glob.glob(f"{prefix}*volatility-greeks*.csv"))
    if not sb_files or not gk_files: return None
    
    df_sb = pd.read_csv(sb_files[-1])
    df_gk = pd.read_csv(gk_files[-1])
    df_sb.columns = [str(c).strip() for c in df_sb.columns]
    df_gk.columns = [str(c).strip() for c in df_gk.columns]
    
    # Strikeをパース
    df_sb['Strike'] = df_sb['Strike'].apply(parse_strike)
    df_gk['Strike'] = df_gk['Strike'].apply(parse_strike)
    
    # 必要なカラムの動的取得
    oi_col = [c for c in df_sb.columns if 'Open Int' in c][0]
    iv_col = [c for c in df_gk.columns if 'IV' in c and '.' not in c][0]
    
    df = pd.merge(df_sb[['Strike', oi_col, f'{oi_col}.1']], 
                  df_gk[['Strike', iv_col, f'{iv_col}.1']], on='Strike')
    
    df['Call_OI'] = df[oi_col].apply(clean_data_string)
    df['Put_OI'] = df[f'{oi_col}.1'].apply(clean_data_string)
    df['Call_IV'] = df[iv_col].apply(clean_data_string) / 100.0
    df['Put_IV'] = df[f'{iv_col}.1'].apply(clean_data_string) / 100.0
    df['IV'] = df[['Call_IV', 'Put_IV']].mean(axis=1)
    return df.sort_values("Strike", ascending=False)

def calculate_gex(df, spot, multiplier):
    T, r = 22/365, 0.045
    iv = np.where(df["IV"] < 0.01, 0.01, df["IV"])
    d1 = (np.log(spot / df["Strike"]) + (r + iv**2 / 2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(T))
    df["Call_GEX"] = (df["Call_OI"] * gamma * spot * multiplier * 0.01) / 1e6
    df["Put_GEX"] = (-df["Put_OI"] * gamma * spot * multiplier * 0.01) / 1e6
    df["Net_GEX"] = df["Call_GEX"] + df["Put_GEX"]
    return df

def export_dashboard(df, spot, filename, config):
    df_zoom = df[(df['Strike'] >= spot * 0.8) & (df['Strike'] <= spot * 1.2)].copy()
    
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, subplot_titles=("GEX Profile", "IV Smile"))
    fig.add_trace(go.Bar(x=df_zoom["Strike"], y=df_zoom["Call_GEX"], name="Call GEX", marker_color="cyan"), row=1, col=1)
    fig.add_trace(go.Bar(x=df_zoom["Strike"], y=df_zoom["Put_GEX"], name="Put GEX", marker_color="magenta"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["Net_GEX"], name="Net GEX", line=dict(color="white")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["IV"]*100, name="IV", line=dict(color="orange"), connectgaps=True), row=2, col=1)
    
    fig.add_vline(x=spot, line_color="yellow")
    fig.update_layout(title=config['name'], template="plotly_dark", height=900)
    
    path = DOCS_DIR / filename
    fig.write_html(path, include_plotlyjs="cdn")
    
    # ナビゲーション追加
    with open(path, 'r', encoding='utf-8') as f: html = f.read()
    nav = '<div style="background:#111; padding:10px; text-align:center;">' + \
          ' | '.join([f'<a href="{c["filename"]}" style="color:white;">{c["name"]}</a>' for c in ASSET_CONFIG.values()]) + '</div>'
    with open(path, 'w', encoding='utf-8') as f: f.write(html.replace('<body>', f'<body>{nav}'))

if __name__ == "__main__":
    DOCS_DIR.mkdir(exist_ok=True)
    for key, config in ASSET_CONFIG.items():
        df = load_barchart_csv(key.lower())
        if df is not None:
            spot = yf.Ticker(config["ticker"]).history(period="1d")['Close'].iloc[-1]
            df = calculate_gex(df, spot, config["multiplier"])
            export_dashboard(df, spot, config["filename"], config)
