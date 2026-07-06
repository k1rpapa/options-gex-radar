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
    spot = tkr.history(period="1d")['Close'].iloc[-1]
    expirations = tkr.options
    if not expirations: raise ValueError(f"No options for {ticker}")
    
    # 建玉合計が最大の限月を選択
    best_expiry = max(expirations[:3], key=lambda e: tkr.option_chain(e).calls['openInterest'].sum())
    chain = tkr.option_chain(best_expiry)
    
    calls = chain.calls[['strike', 'openInterest', 'impliedVolatility']]
    puts = chain.puts[['strike', 'openInterest', 'impliedVolatility']]
    df = pd.merge(calls.rename(columns={'openInterest': 'Call_OI', 'impliedVolatility': 'Call_IV'}),
                  puts.rename(columns={'openInterest': 'Put_OI', 'impliedVolatility': 'Put_IV'}), 
                  on='strike', how='outer').fillna(0)
    df.rename(columns={'strike': 'Strike'}, inplace=True)
    df['IV'] = df[['Call_IV', 'Put_IV']].mean(axis=1)
    return df[(df['Strike'] >= spot * 0.7) & (df['Strike'] <= spot * 1.3)].sort_values("Strike", ascending=False), spot, best_expiry

def calculate_gex(df, spot):
    T = 30 / 365.0
    r = 0.045
    iv = np.where(df["IV"] <= 0, 1e-5, df["IV"])
    d1 = (np.log(spot / df["Strike"]) + (r + iv**2 / 2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(T))
    df["Call_GEX"] = df["Call_OI"] * gamma * spot * 100 * 0.01
    df["Put_GEX"] = -df["Put_OI"] * gamma * spot * 100 * 0.01
    return df

def export_dashboard(df, ticker_name, spot, expiry, output_path):
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4])
    fig.add_trace(go.Bar(y=df["Strike"], x=df["Call_GEX"], name="Call", marker_color="cyan"), row=1, col=1)
    fig.add_trace(go.Bar(y=df["Strike"], x=df["Put_GEX"], name="Put", marker_color="magenta"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Strike"], y=df["IV"]*100, name="IV"), row=2, col=1)
    fig.update_layout(title=f"{ticker_name} | Spot: {spot:.2f}", template="plotly_dark", height=800)
    fig.write_html(output_path, include_plotlyjs="cdn")

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
