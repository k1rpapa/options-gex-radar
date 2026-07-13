import os
import glob
import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy.stats import norm
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import json

# パス定義
ROOT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = ROOT_DIR / "docs"

# 複数アセット定義（S&P 500 E-Mini 追加済）
ASSET_CONFIG = {
    "ES": {"name": "S&P 500 E-Mini (ES)", "ticker": "ES=F", "multiplier": 50, "filename": "es.html"},
    "SI": {"name": "銀先物 (SI)", "ticker": "SI=F", "multiplier": 5000, "filename": "index.html"},
    "NG": {"name": "天然ガス先物 (NG)", "ticker": "NG=F", "multiplier": 10000, "filename": "ng.html"},
    "HG": {"name": "銅先物 (HG)", "ticker": "HG=F", "multiplier": 25000, "filename": "hg.html"},
    "ZS": {"name": "大豆先物 (ZS)", "ticker": "ZS=F", "multiplier": 50, "filename": "zs.html"},
    "ZC": {"name": "コーン先物 (ZC)", "ticker": "ZC=F", "multiplier": 50, "filename": "zc.html"},
    "ZW": {"name": "小麦先物 (ZW)", "ticker": "ZW=F", "multiplier": 50, "filename": "zw.html"}
}

def parse_strike(val):
    s = str(val).split('-')[0].replace('%', '').replace(',', '').replace('s', '').replace('N/A', '0').replace('nan', '0').strip()
    try: return float(s)
    except: return 0.0

def load_barchart_csv(prefix_pattern):
    sb_files = sorted(glob.glob(f"{prefix_pattern}side-by-side*.csv"))
    gk_files = sorted(glob.glob(f"{prefix_pattern}volatility-greeks*.csv"))
    if not sb_files or not gk_files:
        return None, None, None
    sb_path = sb_files[-1]
    gk_path = gk_files[-1]
    
    # ファイル名から日付を抽出 (例: ...-07-12-2026.csv)
    date_match = re.search(r'(\d{2}-\d{2}-\d{4})', sb_path)
    data_date = date_match.group(1) if date_match else "Unknown Date"
    
    return sb_path, gk_path, data_date

def generate_ai_insight(asset_name, spot_price, zero_gamma, call_wall, put_wall, regime):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "<p style='color: red;'>APIキーが設定されていません。GitHub Secretsを確認してください。</p>"
    
    genai.configure(api_key=api_key)
    
    prompt = f"""
    あなたは凄腕のクオンツ・オプション・トレーダーです。以下のGEXデータに基づき、プロのCFDトレーダーに向けた今日のトレードの作戦指令（インサイト）を短く、鋭く、箇条書きで出力してください。Markdownの装飾を効果的に使い、HTMLタグにフォーマットして出力すること。

    # データ
    - 銘柄: {asset_name}
    - 現在価格: {spot_price}
    - ゼロガンマ: {zero_gamma}
    - コールの壁(レジスタンス): {call_wall}
    - プットの壁(サポート): {put_wall}
    - 現在のレジーム: {regime}

    # 出力要件
    初心者向けの解説（GEXとは何か、レジームとは何かなど）は絶対に不要。無駄な前置きや後書きも避けること。
    現在の重力場から読み取れる「具体的なエントリー/エグジット/撤退の目安」だけを、プロフェッショナルなトーンで出力せよ。
    """
    
    try:
        # 動的モデル探索とフォールバックの強靭化
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        preferred_order = [
            "models/gemini-1.5-flash-latest", "models/gemini-1.5-flash", 
            "models/gemini-1.5-pro-latest", "models/gemini-1.5-pro", 
            "models/gemini-pro", "models/gemini-1.0-pro"
        ]
        
        last_error = None
        for target_model in preferred_order:
            if target_model in available_models:
                try:
                    model = genai.GenerativeModel(model_name=target_model.replace("models/", ""))
                    response = model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    last_error = e
                    print(f"[!] Warning: Model {target_model} failed: {e}. Trying next...")
                    continue
        
        raise Exception(f"全てのAIモデルでインサイト生成に失敗しました。詳細: {last_error}")
    except Exception as e:
        return f"<p style='color: #ff4444;'>[エラー] {e}</p>"

def process_asset(asset_key, config):
    print(f"[*] Processing {config['name']}...")
    prefix_pattern = ""
    # CSVファイルのプレフィックスマッチング
    if asset_key == "SI": prefix_pattern = "*si*"
    elif asset_key == "NG": prefix_pattern = "*ng*"
    elif asset_key == "HG": prefix_pattern = "*hg*"
    elif asset_key == "ZS": prefix_pattern = "*zs*"
    elif asset_key == "ZC": prefix_pattern = "*zc*"
    elif asset_key == "ZW": prefix_pattern = "*zw*"
    elif asset_key == "ES": prefix_pattern = "*es*"
    
    sb_path, gk_path, data_date = load_barchart_csv(prefix_pattern)
    if not sb_path or not gk_path:
        print(f"[!] No CSV files found for {asset_key}. Skipping.")
        return
        
    df_sb = pd.read_csv(sb_path)
    df_gk = pd.read_csv(gk_path)
    df_sb.columns = [str(c).strip() for c in df_sb.columns]
    df_gk.columns = [str(c).strip() for c in df_gk.columns]
    
    df_sb['Strike'] = df_sb['Strike'].apply(parse_strike)
    df_gk['Strike'] = df_gk['Strike'].apply(parse_strike)
    
    # 左右に分かれたカラムからCall/Putを動的に特定
    call_oi_col = next((c for c in df_sb.columns if 'Open Int' in c and df_sb.columns.get_loc(c) < list(df_sb.columns).index('Strike')), None)
    put_oi_col = next((c for c in df_sb.columns if 'Open Int' in c and df_sb.columns.get_loc(c) > list(df_sb.columns).index('Strike')), None)
    call_iv_col = next((c for c in df_gk.columns if 'IV' in c and df_gk.columns.get_loc(c) < list(df_gk.columns).index('Strike')), None)
    put_iv_col = next((c for c in df_gk.columns if 'IV' in c and df_gk.columns.get_loc(c) > list(df_gk.columns).index('Strike')), None)
    call_gamma_col = next((c for c in df_gk.columns if 'Gamma' in c and df_gk.columns.get_loc(c) < list(df_gk.columns).index('Strike')), None)
    put_gamma_col = next((c for c in df_gk.columns if 'Gamma' in c and df_gk.columns.get_loc(c) > list(df_gk.columns).index('Strike')), None)
    
    if not all([call_oi_col, put_oi_col, call_iv_col, put_iv_col, call_gamma_col, put_gamma_col]):
        print(f"[!] Missing required columns for {asset_key}.")
        return

    df = pd.merge(df_sb[['Strike', call_oi_col, put_oi_col]], 
                  df_gk[['Strike', call_iv_col, put_iv_col, call_gamma_col, put_gamma_col]], 
                  on='Strike', how='inner')
    
    def clean_val(x):
        s = str(x).split('-')[0].replace('%', '').replace(',', '').replace('s', '').replace('N/A', '0').replace('nan', '0').strip()
        try: return float(s)
        except: return 0.0

    df['Call_OI'] = df[call_oi_col].apply(clean_val)
    df['Put_OI'] = df[put_oi_col].apply(clean_val)
    df['Call_IV'] = df[call_iv_col].apply(clean_val)
    df['Put_IV'] = df[put_iv_col].apply(clean_val)
    df['Call_Gamma'] = df[call_gamma_col].apply(clean_val)
    df['Put_Gamma'] = df[put_gamma_col].apply(clean_val)
    
    # 週末データ欠落対策: period="5d" で取得し最後の有効値を使う
    try:
        spot_history = yf.Ticker(config['ticker']).history(period="5d")
        if spot_history.empty:
            raise ValueError(f"No price data found for {config['ticker']}")
        spot_price = float(spot_history['Close'].iloc[-1])
    except Exception as e:
        print(f"[!] Error fetching spot price for {asset_key}: {e}")
        return
        
    multiplier = config['multiplier']
    
    df['Call_GEX'] = df['Call_OI'] * df['Call_Gamma'] * 100 * multiplier * spot_price
    df['Put_GEX'] = df['Put_OI'] * df['Put_Gamma'] * 100 * multiplier * spot_price * -1
    df['Net_GEX'] = df['Call_GEX'] + df['Put_GEX']
    
    df['IV_Avg'] = (df['Call_IV'] + df['Put_IV']) / 2
    
    # GEXは桁が大きすぎるためミリオン($M)単位に変換
    df['Call_GEX'] /= 1_000_000
    df['Put_GEX'] /= 1_000_000
    df['Net_GEX'] /= 1_000_000
    
    # 描画用の範囲をスポット価格の周辺に限定
    strike_range = spot_price * 0.2
    df_plot = df[(df['Strike'] >= spot_price - strike_range) & (df['Strike'] <= spot_price + strike_range)].copy()
    
    if df_plot.empty:
        print(f"[!] No valid data in strike range for {asset_key}.")
        return

    # Key Levels
    call_wall = df_plot.loc[df_plot['Call_GEX'].idxmax()]['Strike']
    put_wall = df_plot.loc[df_plot['Put_GEX'].idxmin()]['Strike']
    
    # 累積ガンマからZero Gammaを算出
    df_plot_sorted = df_plot.sort_values('Strike')
    cumulative_gex = np.cumsum(df_plot_sorted['Net_GEX'].values[::-1])[::-1] # 上から下へ累積
    zero_gamma_idx = np.argmin(np.abs(cumulative_gex))
    zero_gamma = df_plot_sorted['Strike'].iloc[zero_gamma_idx]
    
    regime = "🟢 POSITIVE GAMMA REGIME (押し目買い優位)" if spot_price > zero_gamma else "🔴 NEGATIVE GAMMA REGIME (パニック売り警戒)"
    
    # AI Insight の生成
    ai_insight_html = generate_ai_insight(config['name'], spot_price, zero_gamma, call_wall, put_wall, regime)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"Dealer Net GEX Profile<br><span style='font-size: 14px; color: {'#00ff00' if spot_price > zero_gamma else '#ff0000'};'>{regime}</span>", "Implied Volatility Smile"))
    
    # Call GEX
    fig.add_trace(go.Bar(x=df_plot['Strike'], y=df_plot['Call_GEX'], name='Call GEX (レジスタンス)', marker_color='#00FFFF', opacity=0.8), row=1, col=1)
    # Put GEX
    fig.add_trace(go.Bar(x=df_plot['Strike'], y=df_plot['Put_GEX'], name='Put GEX (サポート)', marker_color='#FF00FF', opacity=0.8), row=1, col=1)
    # Net GEX
    fig.add_trace(go.Scatter(x=df_plot['Strike'], y=df_plot['Net_GEX'], name='Net GEX', mode='lines+markers', line=dict(color='white', width=2), marker=dict(size=4)), row=1, col=1)
    
    # Zero Gamma Line
    fig.add_vline(x=zero_gamma, line_dash="dashdot", line_color="red", line_width=2,
                  annotation_text=f"Zero-Gamma<br>{zero_gamma}", annotation_position="top left", 
                  annotation_bgcolor="red", annotation_font_color="white", row=1, col=1)
                  
    # Spot Price Line
    fig.add_vline(x=spot_price, line_width=2, line_color="yellow",
                  annotation_text=f"Current Spot<br>{spot_price}", annotation_position="bottom right",
                  annotation_bgcolor="yellow", annotation_font_color="black", row=1, col=1)
                  
    # IV Smile
    fig.add_trace(go.Scatter(x=df_plot['Strike'], y=df_plot['IV_Avg'], name='IV', mode='lines+markers', line=dict(color='#FFA500', width=2), marker=dict(size=4)), row=2, col=1)
    
    fig.update_layout(
        title=f"Quant Options Radar: {config['name']} | Expiry: {Path(sb_path).stem.split('-exp-')[-1].split('-')[0]}<br><span style='font-size: 13px; color: #aaaaaa;'>As of: {data_date}</span>",
        xaxis_title="",
        yaxis_title="GEX ($M)",
        xaxis2_title="Strike Price",
        yaxis2_title="IV (%)",
        template="plotly_dark",
        barmode='relative',
        hovermode='x unified',
        height=800,
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    nav_and_container = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Quant Options Radar - {config['name']}</title>
        <style>
            body {{ background-color: #111111; color: white; font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 0; overflow-x: hidden; }}
            .nav-bar {{ display: flex; flex-wrap: wrap; background-color: #000; padding: 10px; border-bottom: 1px solid #333; }}
            .nav-bar a {{ color: #ccc; text-decoration: none; padding: 8px 15px; margin: 5px; border-radius: 4px; font-size: 14px; font-weight: bold; background-color: #222; transition: all 0.2s; }}
            .nav-bar a:hover {{ background-color: #444; color: white; }}
            .active-nav {{ border: 1px solid #fff; }}
            .chart-container {{ width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }}
            .chart-inner {{ min-width: 800px; }}
            .mobile-notice {{ display: none; text-align: center; font-size: 12px; color: #888; padding: 5px; }}
            .insight-panel {{ margin: 20px auto; max-width: 1200px; padding: 20px; background-color: #1a1a1a; border-left: 4px solid #FFA500; border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            .insight-title {{ color: #FFA500; font-size: 16px; font-weight: bold; margin-bottom: 15px; display: flex; align-items: center; }}
            .insight-title span {{ margin-right: 10px; font-size: 20px; }}
            .insight-content {{ line-height: 1.6; font-size: 14px; }}
            .insight-content h3 {{ color: #ddd; font-size: 15px; border-bottom: 1px solid #333; padding-bottom: 5px; margin-top: 15px; }}
            .insight-content ul {{ padding-left: 20px; }}
            .insight-content li {{ margin-bottom: 8px; }}
            @media (max-width: 768px) {{
                .mobile-notice {{ display: block; }}
                .nav-bar {{ justify-content: center; }}
                .insight-panel {{ margin: 10px; padding: 15px; }}
            }}
        </style>
    </head>
    <body>
    <div class="nav-bar">
        <a href="es.html" style="color: #00BFFF;" {'class="active-nav"' if asset_key == "ES" else ''}>📈 S&P 500 (ES)</a>
        <a href="index.html" style="color: #00FFFF;" {'class="active-nav"' if asset_key == "SI" else ''}>🪙 Silver (SI)</a>
        <a href="ng.html" style="color: #FF00FF;" {'class="active-nav"' if asset_key == "NG" else ''}>🔥 Natural Gas (NG)</a>
        <a href="hg.html" style="color: #FF8C00;" {'class="active-nav"' if asset_key == "HG" else ''}>🥉 Copper (HG)</a>
        <a href="zs.html" style="color: #32CD32;" {'class="active-nav"' if asset_key == "ZS" else ''}>🌱 Soybeans (ZS)</a>
        <a href="zc.html" style="color: #FFD700;" {'class="active-nav"' if asset_key == "ZC" else ''}>🌽 Corn (ZC)</a>
        <a href="zw.html" style="color: #DAA520;" {'class="active-nav"' if asset_key == "ZW" else ''}>🌾 Wheat (ZW)</a>
        <a href="gex_trading_guide.html" style="color: #FFFF00;">📖 Trading Manual</a>
    </div>
    <div class="mobile-notice">📱 グラフを左右にスワイプして詳細を確認できます</div>
    <div class="chart-container">
        <div class="chart-inner">
            {html_str}
        </div>
    </div>
    <div class="insight-panel">
        <div class="insight-title"><span>●</span> DAILY QUANT INSIGHT (Powered by Gemini AI)</div>
        <div class="insight-content">
            {ai_insight_html}
        </div>
    </div>
    </body>
    </html>
    """
    
    out_path = DOCS_DIR / config['filename']
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(nav_and_container)
    print(f"[*] Saved {out_path}")

def main():
    DOCS_DIR.mkdir(exist_ok=True, parents=True)
    with open(DOCS_DIR / ".nojekyll", "w") as f:
        pass
        
    for key, config in ASSET_CONFIG.items():
        try:
            process_asset(key, config)
        except Exception as e:
            print(f"Error: {config['name']} の処理中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
