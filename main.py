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

# 複数アセットの定義とマルチプライヤー設定
ASSET_CONFIG = {
    "SI": {
        "name": "銀先物 (SI)",
        "ticker": "SI=F",
        "multiplier": 5000,
        "filename": "index.html"
    },
    "NG": {
        "name": "天然ガス先物 (NG)",
        "ticker": "NG=F",
        "multiplier": 10000,
        "filename": "ng.html"
    },
    "ZS": {
        "name": "大豆先物 (ZS)",
        "ticker": "ZS=F",
        "multiplier": 5000,
        "filename": "zs.html"
    }
}

def load_barchart_csv(prefix_pattern):
    """プレフィックス（例: si, ng, zs）に合致する最新のCSVペアを読み込む"""
    sb_files = sorted(glob.glob(f"{prefix_pattern}*side-by-side*.csv"))
    gk_files = sorted(glob.glob(f"{prefix_pattern}*volatility-greeks*.csv"))
    
    if not sb_files or not gk_files:
        return None, None
        
    sb_path = sb_files[-1]
    gk_path = gk_files[-1]
    
    # データ読み込み時のフォーマット差異を吸収するため、低レベルで読み込む
    df_sb = pd.read_csv(sb_path)
    df_gk = pd.read_csv(gk_path)
    
    df_sb.columns = [str(c).strip() for c in df_sb.columns]
    df_gk.columns = [str(c).strip() for c in df_gk.columns]
    
    # 銘柄特有の表記揺れ（Strikeが「520-0」のような形式の場合）に対応
    def parse_strike(val):
        s = str(val).split('-')[0].replace(',', '')
        return float(s)

    df_sb['Strike'] = df_sb['Strike'].apply(parse_strike)
    df_gk['Strike'] = df_gk['Strike'].apply(parse_strike)

    oi_candidates = ['Open Int', 'OI', 'Open Interest']
    call_oi_col, put_oi_col = None, None
    for col in oi_candidates:
        if col in df_sb.columns:
            call_oi_col, put_oi_col = col, f"{col}.1"
            break

    iv_candidates = ['IV', 'Implied Vol', 'Implied Volatility']
    call_iv_col, put_iv_col = None, None
    for col in iv_candidates:
        if col in df_gk.columns:
            call_iv_col, put_iv_col = col, f"{col}.1"
            break
            
    if not call_oi_col or not call_iv_col:
        raise KeyError(f"必要なカラムが見つかりません。")
        
    df = pd.merge(df_sb[['Strike', call_oi_col, put_oi_col]], 
                  df_gk[['Strike', call_iv_col, put_iv_col]], 
                  on='Strike', how='inner')
    
    # データクレンジング
    def clean_val(x):
        s = str(x).split('-')[0].replace('%', '').replace(',', '').replace('N/A', '0').replace('nan', '0').strip()
        try: return float(s)
        except: return 0.0

    df['Call_OI'] = df[call_oi_col].apply(clean_val)
    df['Put_OI'] = df[put_oi_col].apply(clean_val)
    df['Call_IV'] = df[call_iv_col].apply(clean_val) / 100.0
    df['Put_IV'] = df[put_iv_col].apply(clean_val) / 100.0
            
    df = df.fillna(0)
    df['IV'] = df[['Call_IV', 'Put_IV']].mean(axis=1)
    
    return df.sort_values("Strike", ascending=False), "Recent"

def fetch_futures_spot(ticker):
    tkr = yf.Ticker(ticker)
    hist = tkr.history(period="1d")
    return hist['Close'].iloc[-1] if not hist.empty else 0.0

def calculate_gex(df, spot, multiplier):
    df = df[(df['Call_OI'] > 0) | (df['Put_OI'] > 0)].copy()
    T = 22 / 365.0 
    r = 0.045
    iv = np.where(df["IV"] <= 0.01, 0.01, df["IV"])
    d1 = (np.log(spot / df["Strike"]) + (r + iv**2 / 2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(T))
    df["Call_GEX"] = (df["Call_OI"] * gamma * spot * multiplier * 0.01) / 1e6
    df["Put_GEX"] = (-df["Put_OI"] * gamma * spot * multiplier * 0.01) / 1e6
    df["Net_GEX"] = df["Call_GEX"] + df["Put_GEX"]
    return df

def extract_flip_point(df, spot):
    df_sorted = df.sort_values("Strike").dropna(subset=["Net_GEX"])
    net_gex = df_sorted["Net_GEX"].values
    strikes = df_sorted["Strike"].values
    sign_flips = np.where(np.diff(np.sign(net_gex)))[0]
    if len(sign_flips) == 0: return np.nan
    closest_flip_idx = min(sign_flips, key=lambda i: abs(strikes[i] - spot))
    x0, x1 = net_gex[closest_flip_idx], net_gex[closest_flip_idx + 1]
    y0, y1 = strikes[closest_flip_idx], strikes[closest_flip_idx + 1]
    return y0 - (x0 * (y1 - y0) / (x1 - x0)) if x1 != x0 else y0

def export_dashboard(df, spot, expiry, output_path, config):
    flip_point = extract_flip_point(df, spot)
    df_zoom = df[(df['Strike'] >= spot * 0.8) & (df['Strike'] <= spot * 1.2)].copy()
    
    # 簡易IVフィルター
    df_zoom["IV_plot"] = np.where(df_zoom["IV"] > df_zoom["IV"].median() * 3, np.nan, df_zoom["IV"])
    
    is_positive = spot > flip_point
    regime_text = "🟢 POSITIVE GAMMA" if is_positive else "🔴 NEGATIVE GAMMA"
    
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
    fig.add_trace(go.Bar(x=df_zoom["Strike"], y=df_zoom["Call_GEX"], name="Call GEX", marker_color="cyan"), row=1, col=1)
    fig.add_trace(go.Bar(x=df_zoom["Strike"], y=df_zoom["Put_GEX"], name="Put GEX", marker_color="magenta"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["Net_GEX"], name="Net GEX", line=dict(color="white")), row=1, col=1)
    
    if not np.isnan(flip_point): fig.add_vline(x=flip_point, line_dash="dash", line_color="red")
    fig.add_vline(x=spot, line_color="yellow")
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["IV_plot"]*100, name="IV", line_color="orange", connectgaps=True), row=2, col=1)
    
    fig.update_layout(title=f"{config['name']} | {regime_text}", template="plotly_dark", height=900)
    fig.write_html(output_path, include_plotlyjs="cdn")

    # ナビゲーションバー追加
    with open(output_path, 'r', encoding='utf-8') as f:
        html = f.read()
    nav = """<div style="background:#111; padding:10px; text-align:center;">
    <a href="index.html" style="color:white; margin:0 10px;">Silver</a>
    <a href="ng.html" style="color:white; margin:0 10px;">Gas</a>
    <a href="zs.html" style="color:white; margin:0 10px;">Soybeans</a></div>"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html.replace('<body>', f'<body>{nav}'))

if __name__ == "__main__":
    DOCS_DIR.mkdir(exist_ok=True)
    for asset_key, config in ASSET_CONFIG.items():
        df, expiry = load_barchart_csv(asset_key.lower())
        if df is not None:
            spot = fetch_futures_spot(config["ticker"])
            df = calculate_gex(df, spot, config["multiplier"])
            export_dashboard(df, spot, expiry, DOCS_DIR / config["filename"], config)
