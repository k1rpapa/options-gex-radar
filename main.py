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
    }
}

def load_barchart_csv(prefix_pattern):
    """プレフィックス（例: si, ng）に合致する最新のCSVペアを読み込む"""
    sb_files = sorted(glob.glob(f"{prefix_pattern}*side-by-side*.csv"))
    gk_files = sorted(glob.glob(f"{prefix_pattern}*volatility-greeks*.csv"))
    
    if not sb_files or not gk_files:
        return None, None
        
    sb_path = sb_files[-1]
    gk_path = gk_files[-1]
    
    print(f"[{prefix_pattern.upper()} Loading] 建玉データ: {sb_path}")
    print(f"[{prefix_pattern.upper()} Loading] ギリシャ・IVデータ: {gk_path}")
    
    df_sb = pd.read_csv(sb_path)
    df_gk = pd.read_csv(gk_path)
    
    df_sb.columns = [str(c).strip() for c in df_sb.columns]
    df_gk.columns = [str(c).strip() for c in df_gk.columns]
    
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
        raise KeyError(f"必要なカラムが見つかりません。OI候補: {call_oi_col}, IV候補: {call_iv_col}")
        
    sb_map = {'Strike': 'Strike', call_oi_col: 'Call_OI', put_oi_col: 'Put_OI'}
    gk_map = {'Strike': 'Strike', call_iv_col: 'Call_IV', put_iv_col: 'Put_IV'}
    
    df_sb_clean = df_sb[['Strike', call_oi_col, put_oi_col]].rename(columns=sb_map)
    df_gk_clean = df_gk[['Strike', call_iv_col, put_iv_col]].rename(columns=gk_map)
        
    df = pd.merge(df_sb_clean, df_gk_clean, on='Strike', how='inner')
    
    def clean_col(series, is_iv=False):
        # 天然ガスの「欠損データ(nan)」も安全に0へ変換する
        s = series.astype(str).str.replace('%', '', regex=False)\
                              .str.replace(',', '', regex=False)\
                              .str.replace('N/A', '0', regex=False)\
                              .str.replace('nan', '0', regex=False)\
                              .str.replace('-', '0', regex=False)\
                              .str.strip()
        s_num = pd.to_numeric(s, errors='coerce').fillna(0)
        return s_num / 100.0 if is_iv else s_num

    df['Call_OI'] = clean_col(df['Call_OI'])
    df['Put_OI'] = clean_col(df['Put_OI'])
    df['Call_IV'] = clean_col(df['Call_IV'], is_iv=True)
    df['Put_IV'] = clean_col(df['Put_IV'], is_iv=True)
            
    df = df.fillna(0)
    df['IV'] = df[['Call_IV', 'Put_IV']].mean(axis=1)
    
    expiry_info = "Unknown"
    if "exp-" in sb_path:
        expiry_info = os.path.basename(sb_path).split("exp-")[1].split("-")[0]
        
    return df.sort_values("Strike", ascending=False), expiry_info

def fetch_futures_spot(ticker):
    tkr = yf.Ticker(ticker)
    hist = tkr.history(period="1d")
    if hist.empty:
        raise ValueError(f"yfinanceから {ticker} のスポット価格取得に失敗しました。")
    return hist['Close'].iloc[-1]

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
    
    if x1 - x0 == 0: return y0
    return y0 - (x0 * (y1 - y0) / (x1 - x0))

def export_dashboard(df, spot, expiry, output_path, config):
    flip_point = extract_flip_point(df, spot)
    
    df_zoom = df[(df['Strike'] >= spot * 0.7) & (df['Strike'] <= spot * 1.3)].copy()
    
    # --- IVノイズフィルター ---
    # 過疎地の極端なIVスパイク（中央値の2.5倍以上）を描画から除外し、スマイル曲線を綺麗に保つ
    iv_median = df_zoom["IV"].median()
    df_zoom["IV_plot"] = np.where(df_zoom["IV"] > iv_median * 2.5, np.nan, df_zoom["IV"])
    
    is_positive = spot > flip_point
    regime_text = "🟢 POSITIVE GAMMA REGIME (押し目買い優位)" if is_positive else "🔴 NEGATIVE GAMMA REGIME (ブレイクアウト・順張り優位)"
    regime_color = "#00FF00" if is_positive else "#FF4444"
    
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.7, 0.3],
        subplot_titles=(f"Dealer Net GEX Profile<br><b style='color:{regime_color}; font-size:16px;'>{regime_text}</b>", "Implied Volatility Smile (Noise Filtered)"),
        shared_xaxes=True, vertical_spacing=0.07
    )
    
    fig.add_trace(go.Bar(x=df_zoom["Strike"], y=df_zoom["Call_GEX"], name="Call GEX (レジスタンス)", marker_color="rgba(0, 255, 255, 0.7)"), row=1, col=1)
    fig.add_trace(go.Bar(x=df_zoom["Strike"], y=df_zoom["Put_GEX"], name="Put GEX (サポート)", marker_color="rgba(255, 0, 255, 0.7)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["Net_GEX"], name="Net GEX", mode="lines+markers", line=dict(color="white", width=2)), row=1, col=1)
    
    if not np.isnan(flip_point):
        fig.add_vline(x=flip_point, line=dict(color="red", width=2, dash="dashdot"), 
                      annotation_text=f"Zero-Gamma<br>{flip_point:.3f}", 
                      annotation_position="top left", 
                      annotation=dict(font=dict(color="white", size=13), bgcolor="rgba(255,0,0,0.6)", bordercolor="red", borderwidth=1),
                      row=1, col=1)

    fig.add_vline(x=spot, line=dict(color="yellow", width=2, dash="solid"), 
                  annotation_text=f"Current Spot<br>{spot:.3f}", 
                  annotation_position="bottom right", 
                  annotation=dict(font=dict(color="black", size=13), bgcolor="rgba(255,255,0,0.8)", bordercolor="yellow", borderwidth=1),
                  row=1, col=1)
        
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["IV_plot"]*100, name="IV", mode="lines+markers", line_color="orange"), row=2, col=1)
    
    fig.update_layout(
        title=f"Quant Options Radar: {config['name']} | Expiry: {expiry}",
        template="plotly_dark", height=950, barmode='relative', hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="GEX ($M)", row=1, col=1)
    fig.update_yaxes(title_text="IV (%)", row=2, col=1)
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)
    
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)

    # --- グローバル・ナビゲーション・バーの注入 ---
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    nav_html = f"""
    <div style="padding: 12px; background-color: #111; text-align: center; border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 9999;">
        <a href="index.html" style="color: #00FFFF; margin: 0 15px; text-decoration: none; font-family: sans-serif; font-weight: bold; font-size: 16px;">🪙 Silver (SI)</a>
        <a href="ng.html" style="color: #FF00FF; margin: 0 15px; text-decoration: none; font-family: sans-serif; font-weight: bold; font-size: 16px;">🔥 Natural Gas (NG)</a>
        <a href="gex_trading_guide.html" style="color: #FFFF00; margin: 0 15px; text-decoration: none; font-family: sans-serif; font-weight: bold; font-size: 16px;">📖 Trading Manual</a>
    </div>
    """
    html_content = html_content.replace('<body>', f'<body style="margin:0;">\n{nav_html}')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / ".nojekyll").touch()
    
    # 全アセットの処理を自動ループ
    for asset_key, config in ASSET_CONFIG.items():
        prefix = asset_key.lower() # 'si' または 'ng'
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
            raise e
