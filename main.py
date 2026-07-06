import os
import yfinance as yf
import pandas as pd
import numpy as np
import time
from scipy.stats import norm
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# main.py の冒頭に追加
import pathlib
ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DOCS_DIR = ROOT_DIR / "docs"

# 実行時
DOCS_DIR.mkdir(exist_ok=True)
file_path = DOCS_DIR / f"{ticker.lower()}.html"

# ターゲットアセット
TARGET_ASSETS = {"UNG": "天然ガス (NG)", "SLV": "銀 (SI)", "CORN": "コーン (ZC)"}

def fetch_proxy_options(ticker, expiry_idx=0):
    tkr = yf.Ticker(ticker)
    spot = tkr.history(period="1d")['Close'].iloc[-1]
    expirations = tkr.options
    if not expirations: raise ValueError(f"No options for {ticker}")
    
    # 建玉が最大の限月を自動選択するロジック
    oi_counts = []
    for exp in expirations[:3]: # 直近3限月で評価
        chain = tkr.option_chain(exp)
        oi_counts.append((exp, chain.calls['openInterest'].sum() + chain.puts['openInterest'].sum()))
    
    best_expiry = max(oi_counts, key=lambda x: x[1])[0]
    chain = tkr.option_chain(best_expiry)
    
    calls = chain.calls[['strike', 'openInterest', 'impliedVolatility']].rename(columns={'openInterest': 'Call_OI', 'impliedVolatility': 'Call_IV'})
    puts = chain.puts[['strike', 'openInterest', 'impliedVolatility']].rename(columns={'openInterest': 'Put_OI', 'impliedVolatility': 'Put_IV'})
    
    df = pd.merge(calls, puts, on='strike', how='outer').fillna(0)
    df.rename(columns={'strike': 'Strike'}, inplace=True)
    df['IV'] = df[['Call_IV', 'Put_IV']].mean(axis=1)
    return df[(df['Strike'] >= spot * 0.7) & (df['Strike'] <= spot * 1.3)].sort_values("Strike", ascending=False), spot, best_expiry

def calculate_gex(df, spot, multiplier=100):
    T = 30 / 365.0
    r = 0.045
    iv = np.where(df["IV"] <= 0, 1e-5, df["IV"])
    d1 = (np.log(spot / df["Strike"]) + (r + iv**2 / 2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(T))
    df["Call_GEX"] = df["Call_OI"] * gamma * spot * multiplier * 0.01
    df["Put_GEX"] = -df["Put_OI"] * gamma * spot * multiplier * 0.01
    return df

def export_dashboard(df, ticker, spot, expiry, output_path):
    call_wall = df.loc[df["Call_GEX"].idxmax(), "Strike"]
    put_wall = df.loc[df["Put_GEX"].idxmin(), "Strike"]
    
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], subplot_titles=("Net GEX", "IV Smile"))
    fig.add_trace(go.Bar(y=df["Strike"], x=df["Call_GEX"], name="Call", marker_color="cyan"), row=1, col=1)
    fig.add_trace(go.Bar(y=df["Strike"], x=df["Put_GEX"], name="Put", marker_color="magenta"), row=1, col=1)
    fig.add_hline(y=call_wall, line=dict(color="cyan", dash="dot"), row=1, col=1)
    fig.add_hline(y=put_wall, line=dict(color="magenta", dash="dot"), row=1, col=1)
    fig.add_hline(y=spot, line=dict(color="yellow", width=2), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Strike"], y=df["IV"]*100, mode="lines+markers", name="IV"), row=2, col=1)
    
    fig.update_layout(title=f"{ticker} | Spot: {spot:.2f}", height=800, template="plotly_dark", showlegend=False)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.write_html(output_path, include_plotlyjs="cdn")

if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    generated = []
    for ticker, name in TARGET_ASSETS.items():
        try:
            df, spot, expiry = fetch_proxy_options(ticker)
            df = calculate_gex(df, spot)
            path = f"docs/{ticker.lower()}.html"
            export_dashboard(df, f"{name} ({ticker})", spot, expiry, path)
            generated.append((name, ticker.lower() + ".html"))
        except Exception as e: print(f"Error {ticker}: {e}")
    
    with open("docs/index.html", "w") as f:
        f.write("<html><body style='background:#111;color:#eee;font-family:sans-serif;'><h2>Radar</h2><ul>")
        for name, link in generated: f.write(f"<li><a href='{link}' style='color:#4db8ff'>{name}</a></li>")
        f.write("</ul></body></html>")
