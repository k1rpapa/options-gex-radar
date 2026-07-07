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
    s = str(val).split('-')[0].replace(',', '').strip()
    try: return float(s)
    except: return 0.0

def load_barchart_csv(prefix_pattern):
    sb_files = sorted(glob.glob(f"{prefix_pattern}*side-by-side*.csv"))
    gk_files = sorted(glob.glob(f"{prefix_pattern}*volatility-greeks*.csv"))
    if not sb_files or not gk_files: return None, None
        
    sb_path = sb_files[-1]
    gk_path = gk_files[-1]
    
    df_sb = pd.read_csv(sb_path)
    df_gk = pd.read_csv(gk_path)
    df_sb.columns = [str(c).strip() for c in df_sb.columns]
    df_gk.columns = [str(c).strip() for c in df_gk.columns]
    
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
        raise KeyError("必要なカラムが見つかりません。")
        
    df = pd.merge(df_sb[['Strike', call_oi_col, put_oi_col]], 
                  df_gk[['Strike', call_iv_col, put_iv_col]], 
                  on='Strike', how='inner')
    
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
    
    expiry_info = "Unknown"
    if "exp-" in sb_path:
        expiry_info = os.path.basename(sb_path).split("exp-")[1].split("-")[0]
        
    return df.sort_values("Strike", ascending=False), expiry_info

def fetch_futures_spot(ticker):
    tkr = yf.Ticker(ticker)
    hist = tkr.history(period="1d")
    if hist.empty: raise ValueError(f"取得失敗: {ticker}")
    return hist['Close'].iloc[-1]

def calculate_gex(df, spot, multiplier):
    df = df[(df['Call_OI'] > 0) | (df['Put_OI'] > 0)].copy()
    T, r = 22 / 365.0, 0.045
    iv = np.where(df["IV"] <= 0.01, 0.01, df["IV"])
    
    d1 = (np.log(spot / df["Strike"]) + (r + iv**2 / 2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(T))
    
    df["Call_GEX"] = (df["Call_OI"] * gamma * spot * multiplier * 0.01) / 1e6
    df["Put_GEX"] = (-df["Put_OI"] * gamma * spot * multiplier * 0.01) / 1e6
    df["Net_GEX"] = df["Call_GEX"] + df["Put_GEX"]
    return df

def extract_flip_point(df, spot):
    df_sorted = df.sort_values("Strike").dropna(subset=["Net_GEX"])
    net_gex, strikes = df_sorted["Net_GEX"].values, df_sorted["Strike"].values
    sign_flips = np.where(np.diff(np.sign(net_gex)))[0]
    if len(sign_flips) == 0: return np.nan
    closest_flip_idx = min(sign_flips, key=lambda i: abs(strikes[i] - spot))
    x0, x1 = net_gex[closest_flip_idx], net_gex[closest_flip_idx + 1]
    y0, y1 = strikes[closest_flip_idx], strikes[closest_flip_idx + 1]
    if x1 - x0 == 0: return y0
    return y0 - (x0 * (y1 - y0) / (x1 - x0))

def export_dashboard(df, spot, expiry, output_path, config):
    flip_point = extract_flip_point(df, spot)
    df_zoom = df[(df['Strike'] >= spot * 0.7) & (df['Strike'] <= spot * 1.3)].copy()
    
    iv_series = df_zoom["IV"].replace(0, np.nan)
    iv_median = iv_series.median()
    df_zoom["IV_plot"] = np.where(iv_series > iv_median * 2.5, np.nan, iv_series)
    
    is_positive = spot > flip_point
    regime_text = "🟢 POSITIVE GAMMA REGIME (押し目買い優位)" if is_positive else "🔴 NEGATIVE GAMMA REGIME (ブレイクアウト・順張り優位)"
    regime_color = "#00FF00" if is_positive else "#FF4444"
    
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.7, 0.3],
        subplot_titles=(f"Dealer Net GEX Profile<br><b style='color:{regime_color}; font-size:16px;'>{regime_text}</b>", "Implied Volatility Smile"),
        shared_xaxes=True, vertical_spacing=0.07
    )
    
    fig.add_trace(go.Bar(x=df_zoom["Strike"], y=df_zoom["Call_GEX"], name="Call GEX (レジスタンス)", marker_color="rgba(0, 255, 255, 0.7)"), row=1, col=1)
    fig.add_trace(go.Bar(x=df_zoom["Strike"], y=df_zoom["Put_GEX"], name="Put GEX (サポート)", marker_color="rgba(255, 0, 255, 0.7)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["Net_GEX"], name="Net GEX", mode="lines+markers", line=dict(color="white", width=2)), row=1, col=1)
    
    if not np.isnan(flip_point):
        fig.add_vline(x=flip_point, line=dict(color="red", width=2, dash="dashdot"), 
                      annotation_text=f"Zero-Gamma<br>{flip_point:.3f}", 
                      annotation_position="top left", 
                      annotation=dict(font=dict(color="white", size=13), bgcolor="rgba(255,0,0,0.6)", bordercolor="red", borderwidth=1), row=1, col=1)

    fig.add_vline(x=spot, line=dict(color="yellow", width=2, dash="solid"), 
                  annotation_text=f"Current Spot<br>{spot:.3f}", 
                  annotation_position="bottom right", 
                  annotation=dict(font=dict(color="black", size=13), bgcolor="rgba(255,255,0,0.8)", bordercolor="yellow", borderwidth=1), row=1, col=1)
        
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["IV_plot"]*100, name="IV", mode="lines+markers", line_color="orange", connectgaps=True), row=2, col=1)
    
    # --- 超重要: グラフ自体のサイズをピクセルで強力に固定 ---
    fig.update_layout(
        title=f"Quant Options Radar: {config['name']} | Expiry: {expiry}",
        template="plotly_dark", 
        width=1200,  # 幅を1200pxに強制固定（スマホが勝手に縮小するのを防ぐ）
        height=850,  # 縦幅も指定
        barmode='relative', hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="GEX ($M)", row=1, col=1)
    fig.update_yaxes(title_text="IV (%)", row=2, col=1)
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)
    
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)

    # --- モバイル完全対応のViewport制御とHTML/CSSの注入 ---
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    custom_head = """
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        body { margin: 0; background-color: #111; font-family: sans-serif; overflow-x: hidden; }
        .nav-bar {
            padding: 10px; background-color: #111; text-align: center; 
            border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 9999;
        }
        .nav-bar a { margin: 0 10px; text-decoration: none; font-weight: bold; font-size: 15px; display: inline-block; padding: 5px; }
        .mobile-notice {
            display: none; background-color: #2c3e50; color: #f1c40f; 
            text-align: center; padding: 8px; font-size: 13px; font-weight: bold;
        }
        .chart-scroll-container {
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch; /* スマホでの滑らかな横スクロール */
        }
        .chart-wrapper {
            margin: 0 auto;
            width: 1200px; /* PCの大画面でグラフを中央寄せにする */
        }
        /* スマホ向けのレイアウト切り替え */
        @media screen and (max-width: 800px) {
            .mobile-notice { display: block; }
            .nav-bar a { font-size: 12px; margin: 0 3px; }
        }
    </style>
    """
    
    nav_and_container = """
    <body>
    <div class="nav-bar">
        <a href="index.html" style="color: #00FFFF;">🪙 Silver (SI)</a>
        <a href="ng.html" style="color: #FF00FF;">🔥 Natural Gas (NG)</a>
        <a href="zs.html" style="color: #32CD32;">🌱 Soybeans (ZS)</a>
        <a href="gex_trading_guide.html" style="color: #FFFF00;">📖 Trading Manual</a>
    </div>
    <div class="mobile-notice">📱 グラフを左右にスワイプして詳細を確認できます</div>
    <div class="chart-scroll-container">
        <div class="chart-wrapper">
    """
    
    html_content = html_content.replace('<head>', f'<head>\n{custom_head}')
    html_content = html_content.replace('<body>', nav_and_container)
    html_content = html_content.replace('</body>', '</div></div></body>')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / ".nojekyll").touch()
    
    for asset_key, config in ASSET_CONFIG.items():
        prefix = asset_key.lower()
        try:
            df, expiry = load_barchart_csv(prefix)
            if df is not None:
                spot = fetch_futures_spot(config["ticker"])
                df = calculate_gex(df, spot, multiplier=config["multiplier"])
                
                output_path = DOCS_DIR / config["filename"]
                export_dashboard(df, spot, expiry, str(output_path), config)
                print(f"[SUCCESS] {config['name']} Dashboard generated at: {output_path}")
            else:
                print(f"[SKIP] {config['name']} のCSVが見つかりません。")
        except Exception as e:
            print(f"[ERROR] {config['name']} の処理中にエラーが発生しました: {e}")
