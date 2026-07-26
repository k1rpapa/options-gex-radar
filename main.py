import os
import glob
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from pathlib import Path
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai

# ==========================================
# 設定とグローバル変数
# ==========================================
ROOT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = ROOT_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

ASSET_CONFIG = {
    "ES": {"name": "🇺🇸 S&P 500 (ES)", "ticker": "ES=F", "multiplier": 50, "filename": "es.html"},
    "NQ": {"name": "💻 NASDAQ 100 (NQ)", "ticker": "NQ=F", "multiplier": 20, "filename": "nq.html"},
    "SI": {"name": "🥈 シルバー (SI)", "ticker": "SI=F", "multiplier": 5000, "filename": "index.html"},
    "CL": {"name": "🛢️ 原油 (CL)", "ticker": "CL=F", "multiplier": 1000, "filename": "cl.html"},
    "PL": {"name": "✨ プラチナ (PL)", "ticker": "PL=F", "multiplier": 50, "filename": "pl.html"},
    "NG": {"name": "🔥 天然ガス (NG)", "ticker": "NG=F", "multiplier": 10000, "filename": "ng.html"},
    "HG": {"name": "🧱 銅 (HG)", "ticker": "HG=F", "multiplier": 25000, "filename": "hg.html"},
    "ZS": {"name": "🌱 大豆 (ZS)", "ticker": "ZS=F", "multiplier": 50, "filename": "zs.html"},
    "ZC": {"name": "🌽 トウモロコシ (ZC)", "ticker": "ZC=F", "multiplier": 50, "filename": "zc.html"},
    "ZW": {"name": "🌾 小麦 (ZW)", "ticker": "ZW=F", "multiplier": 50, "filename": "zw.html"},
    "SB": {"name": "🍬 砂糖 (SB)", "ticker": "SB=F", "multiplier": 1120, "filename": "sb.html"},
    "HE": {"name": "🐷 豚肉 (HE)", "ticker": "HE=F", "multiplier": 400, "filename": "he.html"}
}

# ==========================================
# ヘルパー関数 (堅牢なデータ抽出)
# ==========================================
def parse_strike(val):
    s = str(val).split('-')[0].replace(',', '').replace('s', '').strip()
    try:
        return float(s)
    except:
        return 0.0

def clean_val(val):
    s = str(val).replace(',', '').replace('s', '').replace('%', '').strip()
    if s == 'N/A' or s == '':
        return 0.0
    try:
        return float(s)
    except:
        return 0.0

def load_barchart_csv(asset_key):
    prefix = asset_key.lower()
    sb_files = sorted(glob.glob(f"{prefix}*side-by-side*.csv"))
    gk_files = sorted(glob.glob(f"{prefix}*volatility-greeks*.csv"))
    
    if not sb_files or not gk_files:
        return None, None, None, None
        
    sb_path = sb_files[-1]
    gk_path = gk_files[-1]
    
    df_sb = pd.read_csv(sb_path)
    df_gk = pd.read_csv(gk_path)
    
    # 満期日の抽出
    match = re.search(r'exp-(\d{2}_\d{2}_\d{2})', sb_path)
    expiry = match.group(1) if match else "Unknown"
    
    # ファイル名から「データ取得日（As of）」を正確に抽出
    date_match = re.search(r'-(\d{2}-\d{2}-\d{4})\.csv', sb_path)
    as_of_date = date_match.group(1) if date_match else None
    
    return df_sb, df_gk, expiry, as_of_date

def generate_batched_insights(asset_summaries):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {k: "<p style='color:#fe8983;'>APIキー未設定</p>" for k in asset_summaries.keys()}
        
    genai.configure(api_key=api_key)
    # 2.5-flashをメインとし、フォールバックとして確実なlatest系を指定
    models_to_try = ["gemini-2.5-flash", "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
    
    prompt = f"""
あなたは金融工学とオプション取引に精通した「リード・クオンツアナリスト」です。
以下の最新のGEX（ガンマ・エクスポージャー）データに基づき、オプション初心者にも分かるように分析とトレード戦略を出力してください。

【全銘柄のデータ】
{json.dumps(asset_summaries, ensure_ascii=False, indent=2)}

【ルール】
- Web埋め込み用HTML断片のみを出力。
- <h3>, <ul>, <li>, <p>, <strong>タグを使用。
- 各銘柄のレジームに基づく「エントリー、利確、損切り」のアクションプランを提示。
- Markdown記法は一切禁止。

【JSONフォーマット (厳守)】
{{
  "ES": "<h3>🇺🇸 S&P 500 (ES) 分析</h3><ul><li>...</li></ul>",
  ...
}}
"""
    
    for target_model in models_to_try:
        try:
            model = genai.GenerativeModel(model_name=target_model)
            response = model.generate_content(prompt)
            res_text = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
            return json.loads(res_text)
        except Exception as e:
            print(f"[-] AI Model {target_model} failed: {e}")
            
    return {k: "<p style='color:#fe8983;'>インサイト生成エラー</p>" for k in asset_summaries.keys()}

def process_asset_data(asset_key, config):
    df_sb, df_gk, expiry, file_as_of = load_barchart_csv(asset_key)
    if df_sb is None:
        raise FileNotFoundError(f"CSV files not found for {asset_key}")
        
    df_sb.columns = [str(c).strip() for c in df_sb.columns]
    df_gk.columns = [str(c).strip() for c in df_gk.columns]
    
    df_sb['Strike'] = df_sb['Strike'].apply(parse_strike)
    df_gk['Strike'] = df_gk['Strike'].apply(parse_strike)
    
    # データの安全な抽出
    oi_idx = [i for i, col in enumerate(df_sb.columns) if 'Open Int' in col or 'OI' in col]
    df_sb['Call_OpenInt'] = df_sb.iloc[:, oi_idx[0]].apply(clean_val) if len(oi_idx)>=2 else 0.0
    df_sb['Put_OpenInt'] = df_sb.iloc[:, oi_idx[1]].apply(clean_val) if len(oi_idx)>=2 else 0.0

    gamma_idx = [i for i, col in enumerate(df_gk.columns) if 'Gamma' in col]
    df_gk['Gamma_Call'] = df_gk.iloc[:, gamma_idx[0]].apply(clean_val) if len(gamma_idx)>=2 else 0.0
    df_gk['Gamma_Put'] = df_gk.iloc[:, gamma_idx[1]].apply(clean_val) if len(gamma_idx)>=2 else 0.0

    iv_idx = [i for i, col in enumerate(df_gk.columns) if 'IV' in col and 'Skew' not in col]
    df_gk['IV_Call'] = df_gk.iloc[:, iv_idx[0]].apply(clean_val) if len(iv_idx)>=2 else 0.0
    df_gk['IV_Put'] = df_gk.iloc[:, iv_idx[1]].apply(clean_val) if len(iv_idx)>=2 else 0.0

    df_merged = df_gk.groupby('Strike').max().merge(df_sb.groupby('Strike').sum(), on='Strike', how='outer').fillna(0)
                         
    mult = config['multiplier']
    df_merged['Call_GEX'] = df_merged['Gamma_Call'] * df_merged['Call_OpenInt'] * mult * 100 / 1e6 
    df_merged['Put_GEX'] = df_merged['Gamma_Put'] * df_merged['Put_OpenInt'] * mult * 100 * -1 / 1e6 
    df_merged['Total_GEX'] = df_merged['Call_GEX'] + df_merged['Put_GEX']
    
    spot_price = 0.0
    try:
        hist = yf.Ticker(config['ticker']).history(period="5d")
        if not hist.empty: spot_price = float(hist['Close'].iloc[-1])
    except: pass
    if spot_price == 0.0: spot_price = df_merged.index.median()

    # ファイルの日付を最優先し、なければ現在日
    spot_date = file_as_of if file_as_of else datetime.now().strftime('%m-%d-%Y')

    df_filtered = df_merged[(df_merged.index >= spot_price * 0.8) & (df_merged.index <= spot_price * 1.2)]
    
    # ゼロガンマ算出
    zero_gamma_strike = spot_price
    if not df_filtered.empty:
        df_valid = df_filtered[df_filtered['Call_OpenInt'] + df_filtered['Put_OpenInt'] > 0]
        if not df_valid.empty:
            zero_gamma_strike = df_valid['Total_GEX'].abs().idxmin()

    # 壁の抽出
    call_walls = df_filtered.nlargest(2, 'Call_GEX')[['Call_GEX']].to_dict('index')
    put_walls = df_filtered.nsmallest(2, 'Put_GEX')[['Put_GEX']].to_dict('index')
    
    data_summary = {
        "asset_name": config['name'], "spot": round(spot_price, 3),
        "zero_gamma": round(float(zero_gamma_strike), 2),
        "call_walls": [{"Strike": k, "GEX": v} for k, v in call_walls.items()],
        "put_walls": [{"Strike": k, "GEX": v} for k, v in put_walls.items()]
    }
    
    # プロット
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Bar(x=df_filtered.index, y=df_filtered['Call_GEX'], name='Call GEX', marker_color='#06bbdf'), row=1, col=1)
    fig.add_trace(go.Bar(x=df_filtered.index, y=df_filtered['Put_GEX'], name='Put GEX', marker_color='#c598ff'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Total_GEX'], name='Net GEX', line=dict(color='white')), row=1, col=1)
    fig.add_vline(x=spot_price, line_color="yellow")
    fig.add_vline(x=zero_gamma_strike, line_color="red", line_dash="dash")
    
    fig.update_layout(title=f"Quant Options Radar: {config['name']} | Expiry: {expiry}<br><sub>As of: {spot_date}</sub>", template="plotly_dark", paper_bgcolor="#101218", plot_bgcolor="#101218")
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn'), data_summary, expiry

def main():
    graphs, asset_summaries = {}, {}
    for key, config in ASSET_CONFIG.items():
        try:
            graph_html, summary, _ = process_asset_data(key, config)
            graphs[key] = graph_html
            asset_summaries[key] = summary
        except Exception as e:
            print(f"[-] Error {key}: {e}")

    ai_insights = generate_batched_insights(asset_summaries)

    for key, config in ASSET_CONFIG.items():
        html_content = f"""
        <!DOCTYPE html><html lang="ja" data-theme="dark"><head><meta charset="UTF-8"><title>{config['name']}</title>
        <style>body{{background:#101218;color:#fff;font-family:sans-serif;margin:0;}} .nav-tabs{{background:#1a1d21;padding:10px;display:flex;gap:10px;}} .active{{background:#0b57d0;color:white;padding:5px 10px;}} .container{{max-width:1400px;margin:auto;padding:20px;}}</style></head>
        <body><div class="nav-tabs">{''.join([f'<a href="{cfg["filename"]}" class="{"active" if k == key else ""}">{cfg["name"]}</a>' for k, cfg in ASSET_CONFIG.items()])}</div>
        <div class="container">{graphs.get(key, "")}<div class="ai-panel">{ai_insights.get(key, "")}</div></div></body></html>
        """
        with open(DOCS_DIR / config['filename'], "w", encoding="utf-8") as f: f.write(html_content)

if __name__ == "__main__":
    main()
