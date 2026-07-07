import os
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# パス定義の定数化
ROOT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = ROOT_DIR / "docs"

TARGET_ASSETS = {"UNG": "天然ガス (NG)", "SLV": "銀 (SI)", "CORN": "コーン (ZC)"}

def fetch_proxy_options(ticker):
    tkr = yf.Ticker(ticker)
    # 履歴から最新のスポット価格を取得
    hist = tkr.history(period="5d")
    if hist.empty: raise ValueError(f"No price data for {ticker}")
    spot = hist['Close'].iloc[-1]
    
    expirations = tkr.options
    if not expirations: raise ValueError(f"No options for {ticker}")
    
    # 複数の限月を走査し、OI合計が最大のものを選択
    best_expiry = max(expirations[:5], key=lambda e: tkr.option_chain(e).calls['openInterest'].sum() + tkr.option_chain(e).puts['openInterest'].sum())
    chain = tkr.option_chain(best_expiry)
    
    # データクリーニング: 無効なストライクやゼロOIを除去
    df = pd.merge(chain.calls.rename(columns={'openInterest': 'Call_OI', 'impliedVolatility': 'Call_IV'}),
                  chain.puts.rename(columns={'openInterest': 'Put_OI', 'impliedVolatility': 'Put_IV'}), 
                  on='strike', how='outer').fillna(0)
    
    df = df[(df['strike'] > spot * 0.5) & (df['strike'] < spot * 1.5)] # 現物価格の±50%に限定
    df['IV'] = df[['Call_IV', 'Put_IV']].mean(axis=1)
    return df.rename(columns={'strike': 'Strike'}), spot, best_expiry

def calculate_gex(df, spot, multiplier=100):
    # 無効データ（OIがゼロのストライク）を排除し、ガンマの暴走を防ぐ
    df = df[(df['Call_OI'] > 0) | (df['Put_OI'] > 0)].copy()

    T = 30 / 365.0
    r = 0.045
    # IVが極端に低い場合のゼロ除算/無限大発散を防止
    iv = np.where(df["IV"] <= 0.01, 0.01, df["IV"]) 
    
    d1 = (np.log(spot / df["Strike"]) + (r + iv**2 / 2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(T))
    
    # 単位を「$ Millions per 1% move」にスケーリングして視認性を劇的に向上
    df["Call_GEX"] = (df["Call_OI"] * gamma * spot * multiplier * 0.01) / 1e6
    df["Put_GEX"] = (-df["Put_OI"] * gamma * spot * multiplier * 0.01) / 1e6
    df["Net_GEX"] = df["Call_GEX"] + df["Put_GEX"]
    
    return df

def extract_flip_point(df):
    """Net GEXがゼロを跨ぐストライク（Zero-Gamma Flip）を線形補間で特定"""
    df_sorted = df.sort_values("Strike").dropna(subset=["Net_GEX"])
    net_gex = df_sorted["Net_GEX"].values
    strikes = df_sorted["Strike"].values
    
    sign_flips = np.where(np.diff(np.sign(net_gex)))[0]
    if len(sign_flips) == 0: return np.nan
        
    max_oi_idx = df_sorted[['Call_OI', 'Put_OI']].sum(axis=1).idxmax()
    center_strike = df_sorted.loc[max_oi_idx, "Strike"]
    closest_flip_idx = min(sign_flips, key=lambda i: abs(strikes[i] - center_strike))
    
    x0, x1 = net_gex[closest_flip_idx], net_gex[closest_flip_idx + 1]
    y0, y1 = strikes[closest_flip_idx], strikes[closest_flip_idx + 1]
    if x1 - x0 == 0: return y0
        
    return y0 - (x0 * (y1 - y0) / (x1 - x0))
    
def export_dashboard(df, ticker_name, spot, expiry, output_path):
    flip_point = extract_flip_point(df)
    
    # 構造改革: X軸を共有 (shared_xaxes=True) して縦に並べる
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.7, 0.3], 
        subplot_titles=("Dealer Net GEX Profile ($ Millions / 1% Move)", "Implied Volatility Smile"),
        shared_xaxes=True, vertical_spacing=0.05
    )
    
    # GEX Profile (縦向きのバーに変更し、X軸をStrikeに統一)
    fig.add_trace(go.Bar(x=df["Strike"], y=df["Call_GEX"], name="Call GEX", marker_color="rgba(0, 255, 255, 0.6)"), row=1, col=1)
    fig.add_trace(go.Bar(x=df["Strike"], y=df["Put_GEX"], name="Put GEX", marker_color="rgba(255, 0, 255, 0.6)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Strike"], y=df["Net_GEX"], name="Net GEX", mode="lines+markers", line=dict(color="white", width=2)), row=1, col=1)
    
    # Key Levels (SpotとFlip Pointを垂直線に変更)
    fig.add_vline(x=spot, line=dict(color="yellow", width=2, dash="solid"), annotation_text=f"Spot: {spot:.2f}", annotation_position="top left", row=1, col=1)
    if not np.isnan(flip_point):
        fig.add_vline(x=flip_point, line=dict(color="red", width=2, dash="dashdot"), annotation_text=f"Zero-Gamma: {flip_point:.2f}", annotation_position="top right", row=1, col=1)
        
    # IV Smile
    fig.add_trace(go.Scatter(x=df["Strike"], y=df["IV"]*100, name="IV", mode="lines+markers", line_color="orange"), row=2, col=1)
    if not np.isnan(flip_point):
        fig.add_vline(x=flip_point, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
        
    # レイアウトの最適化 (hovermode='x unified' で上下同時にデータを確認可能に)
    fig.update_layout(
        title=f"{ticker_name} | Expiry: {expiry}", 
        template="plotly_dark", height=900, barmode='relative', 
        hovermode='x unified', showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Y軸のラベル設定
    fig.update_yaxes(title_text="GEX ($M)", row=1, col=1)
    fig.update_yaxes(title_text="IV (%)", row=2, col=1)
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)
    
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)

if __name__ == "__main__":
    DOCS_DIR.mkdir(exist_ok=True)
    generated = []
    for ticker, name in TARGET_ASSETS.items():
        try:
            df, spot, expiry = fetch_proxy_options(ticker)
            df = calculate_gex(df, spot)
            path = DOCS_DIR / f"{ticker.lower()}.html"
            export_dashboard(df, f"{name} ({ticker})", spot, expiry, str(path))
            generated.append((name, f"{ticker.lower()}.html"))
            print(f"Generated: {path}")
        except Exception as e: print(f"Error {ticker}: {e}")
    
    with open(DOCS_DIR / "index.html", "w") as f:
        f.write("<html><body style='background:#111;color:#eee;'><h2>Radar</h2><ul>")
        for name, link in generated: f.write(f"<li><a href='{link}'>{name}</a></li>")
        f.write("</ul></body></html>")
