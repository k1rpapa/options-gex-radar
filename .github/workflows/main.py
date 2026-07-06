import os
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Ingestion Layer (yfinance Proxy) ---
def fetch_proxy_options(ticker="UNG", expiry_idx=0):
    tkr = yf.Ticker(ticker)
    spot = tkr.history(period="1d")['Close'].iloc[-1]
    
    expirations = tkr.options
    if not expirations:
        raise ValueError(f"No options data for {ticker}")
        
    expiry = expirations[expiry_idx]
    chain = tkr.option_chain(expiry)
    
    calls = chain.calls[['strike', 'openInterest', 'impliedVolatility']].rename(
        columns={'openInterest': 'Call_OI', 'impliedVolatility': 'Call_IV'})
    puts = chain.puts[['strike', 'openInterest', 'impliedVolatility']].rename(
        columns={'openInterest': 'Put_OI', 'impliedVolatility': 'Put_IV'})
        
    df = pd.merge(calls, puts, on='strike', how='outer').fillna(0)
    df.rename(columns={'strike': 'Strike'}, inplace=True)
    df['IV'] = df[['Call_IV', 'Put_IV']].mean(axis=1)
    
    # フィルタリング (Spot ± 30%)
    df = df[(df['Strike'] >= spot * 0.7) & (df['Strike'] <= spot * 1.3)].copy()
    return df.sort_values("Strike", ascending=False), spot, expiry

# --- 2. Compute Layer (Vectorized BS Engine) ---
def calculate_gex_profile(df, spot, multiplier=100):
    T = 30 / 365.0
    r = 0.045
    iv_safe = np.where(df["IV"] <= 0, 1e-5, df["IV"])
    
    d1 = (np.log(spot / df["Strike"]) + (r + (iv_safe ** 2) / 2) * T) / (iv_safe * np.sqrt(T))
    gamma = norm.pdf(d1) / (spot * iv_safe * np.sqrt(T))
    
    df["Call_GEX"] = df["Call_OI"] * gamma * spot * multiplier * 0.01
    df["Put_GEX"] = -df["Put_OI"] * gamma * spot * multiplier * 0.01
    return df

def extract_thresholds(df):
    strikes = df['Strike'].values
    total_oi = (df['Call_OI'] + df['Put_OI']).values
    
    peaks, _ = find_peaks(total_oi, prominence=np.max(total_oi) * 0.1)
    pin = strikes[peaks[np.argmax(total_oi[peaks])]] if len(peaks) > 0 else np.nan
    call_wall = strikes[np.argmax(df['Call_OI'].values)]
    put_wall = strikes[np.argmax(df['Put_OI'].values)]
    
    return {"Pin_Strike": pin, "Call_Wall": call_wall, "Put_Wall": put_wall}

# --- 3. Presentation Layer (Static HTML Export) ---
def export_dashboard(df, ticker, spot, expiry, thresholds, output_path="docs/index.html"):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.1,
        row_heights=[0.6, 0.4], subplot_titles=("Dealer Gamma Exposure ($ / 1% Move)", "Implied Volatility Smile")
    )
    
    # GEX Bar Chart
    fig.add_trace(go.Bar(y=df["Strike"], x=df["Call_GEX"], name="Call GEX", orientation="h", marker_color="rgba(31, 119, 180, 0.8)"), row=1, col=1)
    fig.add_trace(go.Bar(y=df["Strike"], x=df["Put_GEX"], name="Put GEX", orientation="h", marker_color="rgba(214, 39, 40, 0.8)"), row=1, col=1)
    
    # Thresholds
    fig.add_hline(y=thresholds["Call_Wall"], line_dash="dot", line_color="cyan", annotation_text="Call Wall", row=1, col=1)
    fig.add_hline(y=thresholds["Put_Wall"], line_dash="dot", line_color="magenta", annotation_text="Put Wall", row=1, col=1)
    fig.add_hline(y=spot, line_solid="solid", line_color="yellow", line_width=2, annotation_text="Spot", row=1, col=1)
    
    # IV Smile
    fig.add_trace(go.Scatter(x=df["Strike"], y=df["IV"] * 100, mode="lines+markers", name="IV", line_color="rgba(255, 127, 14, 0.9)"), row=2, col=1)
    
    fig.update_layout(
        title=f"Quant Options Radar: {ticker} (Expiry: {expiry})<br><sup>Spot: ${spot:.2f} | Generated via CI/CD</sup>",
        barmode="relative", height=900, template="plotly_dark",
        margin=dict(l=10, r=10, t=80, b=20), showlegend=False
    )
    fig.update_yaxes(title_text="Strike Price", autorange="reversed", row=1, col=1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True, default_width="100%", default_height="100%")
    print(f"[SUCCESS] Dashboard exported to {output_path}")

# --- Orchestrator ---
if __name__ == "__main__":
    ticker = "UNG" # 天然ガスのプロキシETF
    df, spot, expiry = fetch_proxy_options(ticker)
    df = calculate_gex_profile(df, spot)
    thresholds = extract_thresholds(df)
    export_dashboard(df, ticker, spot, expiry, thresholds)
