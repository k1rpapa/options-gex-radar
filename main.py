<!-- ... existing code ... -->
import os
import glob
import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy.stats import norm
import yfinance as yf
import plotly.graph_objects as go
<!-- ... existing code ... -->
def parse_strike(val):
    s = str(val).split('-')[0].replace(',', '').strip()
    try: return float(s)
    except: return 0.0

def load_barchart_csv(prefix_pattern):
    sb_files = sorted(glob.glob(f"{prefix_pattern}*side-by-side*.csv"))
    gk_files = sorted(glob.glob(f"{prefix_pattern}*volatility-greeks*.csv"))
    if not sb_files or not gk_files: return None, None, None
        
    sb_path = sb_files[-1]
    gk_path = gk_files[-1]
<!-- ... existing code ... -->
    df['Call_IV'] = df[call_iv_col].apply(clean_val) / 100.0
    df['Put_IV'] = df[put_iv_col].apply(clean_val) / 100.0
            
    df = df.fillna(0)
    df['IV'] = df[['Call_IV', 'Put_IV']].mean(axis=1)
    
    expiry_info = "Unknown"
    data_date = "Unknown"
    basename = os.path.basename(sb_path)
    
    if "exp-" in basename:
        expiry_info = basename.split("exp-")[1].split("-")[0]
        
    # ファイル名末尾の MM-DD-YYYY 形式の日付を抽出
    date_match = re.search(r'(\d{2}-\d{2}-\d{4})\.csv$', basename)
    if date_match:
        data_date = date_match.group(1)
        
    return df.sort_values("Strike", ascending=False), expiry_info, data_date

def fetch_futures_spot(ticker):
<!-- ... existing code ... -->
    if x1 - x0 == 0: return y0
    return y0 - (x0 * (y1 - y0) / (x1 - x0))

def export_dashboard(df, spot, expiry, data_date, output_path, config):
    flip_point = extract_flip_point(df, spot)
    df_zoom = df[(df['Strike'] >= spot * 0.7) & (df['Strike'] <= spot * 1.3)].copy()
<!-- ... existing code ... -->
    fig.add_vline(x=spot, line=dict(color="yellow", width=2, dash="solid"), 
                  annotation_text=f"Current Spot<br>{spot:.3f}", 
                  annotation_position="bottom right", 
                  annotation=dict(font=dict(color="black", size=13), bgcolor="rgba(255,255,0,0.8)", bordercolor="yellow", borderwidth=1), row=1, col=1)
        
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["IV_plot"]*100, name="IV", mode="lines+markers", line_color="orange", connectgaps=True), row=2, col=1)
    
    # --- 改善ポイント1: Plotly側での固定幅指定を解除し、上部余白を広げて文字被りを解消 ---
    fig.update_layout(
        title=f"Quant Options Radar: {config['name']} | Expiry: {expiry}<br><span style='font-size: 13px; color: #aaaaaa;'>As of: {data_date}</span>",
        template="plotly_dark", 
        height=850,  # 縦幅だけを指定。横幅はCSSに任せる
        margin=dict(t=120),  # 上部の余白(Top Margin)を広げて凡例との被りを防ぐ
<!-- ... existing code ... -->
if __name__ == "__main__":
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / ".nojekyll").touch()
    
    for asset_key, config in ASSET_CONFIG.items():
        prefix = asset_key.lower()
        try:
            df, expiry, data_date = load_barchart_csv(prefix)
            if df is not None:
                spot = fetch_futures_spot(config["ticker"])
                df = calculate_gex(df, spot, multiplier=config["multiplier"])
                
                output_path = DOCS_DIR / config["filename"]
                export_dashboard(df, spot, expiry, data_date, str(output_path), config)
                print(f"[SUCCESS] {config['name']} Dashboard generated at: {output_path}")
            else:
                print(f"[SKIP] {config['name']} のCSVが見つかりません。")
        except Exception as e:
            print(f"[ERROR] {config['name']} の処理中にエラーが発生しました: {e}")
