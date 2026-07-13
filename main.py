import os
import glob
import time
import traceback
import pandas as pd
import numpy as np
import re
from pathlib import Path
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai

ROOT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = ROOT_DIR / "docs"

# 複数アセット定義（全7銘柄・マルチプライヤー修正済）
ASSET_CONFIG = {
    "ES": {"name": "S&P 500 (ES)", "ticker": "ES=F", "multiplier": 50, "filename": "es.html"},
    "SI": {"name": "シルバー (SI)", "ticker": "SI=F", "multiplier": 5000, "filename": "index.html"},
    "NG": {"name": "天然ガス (NG)", "ticker": "NG=F", "multiplier": 10000, "filename": "ng.html"},
    "HG": {"name": "銅 (HG)", "ticker": "HG=F", "multiplier": 25000, "filename": "hg.html"},
    "ZS": {"name": "大豆 (ZS)", "ticker": "ZS=F", "multiplier": 50, "filename": "zs.html"},
    "ZC": {"name": "トウモロコシ (ZC)", "ticker": "ZC=F", "multiplier": 50, "filename": "zc.html"},
    "ZW": {"name": "小麦 (ZW)", "ticker": "ZW=F", "multiplier": 50, "filename": "zw.html"}
}

def clean_val(val):
    if pd.isna(val) or val == 'N/A':
        return 0.0
    val_str = str(val).replace(',', '').replace('%', '').replace('s', '').strip()
    try:
        return float(val_str)
    except:
        return 0.0

def parse_strike(val):
    s = str(val).split('-')[0].replace(',', '').replace('s', '').strip()
    try:
        return float(s)
    except:
        return 0.0

def get_col_data(df, primary_name, secondary_name):
    """
    Pandasのマージ仕様変更に強い、安全なカラム抽出ヘルパー。
    存在すればそのSeriesを返し、無ければゼロ埋めのSeriesを返す。
    """
    if primary_name in df.columns:
        return df[primary_name]
    if secondary_name in df.columns:
        return df[secondary_name]
    return pd.Series([0.0] * len(df))

def load_barchart_csv(asset_key):
    """
    Barchartのファイル名揺れを吸収するため、シンプルに先頭一致とキーワードで検索
    """
    prefix = "esu" if asset_key == "ES" else asset_key.lower()
    
    sb_files = sorted(glob.glob(f"{prefix}*side-by-side*.csv"))
    gk_files = sorted(glob.glob(f"{prefix}*volatility-greeks*.csv"))
    
    if not sb_files or not gk_files:
        return None, None, None, "Unknown"
        
    sb_path = sb_files[-1]
    gk_path = gk_files[-1]
    
    date_match = re.search(r'-(\d{2}-\d{2}-\d{4})\.csv$', sb_path)
    file_date = date_match.group(1) if date_match else "Unknown"
    
    expiry_match = re.search(r'-exp-(.*?)-show', sb_path)
    expiry_str = expiry_match.group(1) if expiry_match else "Unknown"
    
    df_sb = pd.read_csv(sb_path)
    df_gk = pd.read_csv(gk_path)
    df_sb.columns = [str(c).strip() for c in df_sb.columns]
    df_gk.columns = [str(c).strip() for c in df_gk.columns]
    
    df_sb['Strike'] = df_sb['Strike'].apply(parse_strike)
    df_gk['Strike'] = df_gk['Strike'].apply(parse_strike)
    
    return df_sb, df_gk, file_date, expiry_str

def generate_ai_insight(asset_name, spot_price, zero_gamma, call_wall, put_wall, regime):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "<p style='color: red;'>APIキーが設定されていません。GitHub Secretsを確認してください。</p>"
    
    genai.configure(api_key=api_key)
    
    prompt = f"""
    あなたは凄腕のクオンツ・オプション・トレーダーです。以下のGEXデータに基づき、プロのCFDトレーダーに向けた今日のトレードの作戦指令（インサイト）を出力してください。

    # データ
    - 銘柄: {asset_name}
    - 現在価格: {spot_price}
    - ゼロガンマ: {zero_gamma}
    - コールの壁(レジスタンス): {call_wall}
    - プットの壁(サポート): {put_wall}
    - 現在のレジーム: {regime}

    # 出力要件 (厳守事項)
    1. 初心者向けの解説や、無駄な前置き・挨拶は一切不要。
    2. Markdownのコードブロック記号（``` や ```html など）は絶対に出力しないこと。
    3. 以下のHTMLテンプレート構造に**完全に**従い、中身のテキストのみを状況に合わせて書き換えて出力すること。ダッシュボードの統一感を保つため、タグの構造やインラインCSSは一切変更しないこと。

    # 出力HTMLテンプレート
    <div>
        <h4 style="color: #00FFFF; margin-top: 0; border-bottom: 1px solid #444; padding-bottom: 5px; font-size: 15px;">■ オペレーション指令: {asset_name}</h4>
        <ul style="list-style-type: none; padding-left: 0; margin-bottom: 0;">
            <li style="margin-bottom: 12px;">
                <span style="background-color: #444; padding: 2px 6px; border-radius: 3px; color: #fff; font-weight: bold; font-size: 12px;">現状認識</span>
                <span style="margin-left: 5px;">（ここに重力場とレジームの現状を簡潔に1〜2文で記述。各壁の数値を必ず含めること。）</span>
            </li>
            <li style="margin-bottom: 12px;">
                <span style="background-color: #0055ff; padding: 2px 6px; border-radius: 3px; color: #fff; font-weight: bold; font-size: 12px;">ロング戦略</span>
                <span style="margin-left: 5px;">（ここに具体的なエントリーポイント、利確目標、撤退ラインを記述。）</span>
            </li>
            <li style="margin-bottom: 12px;">
                <span style="background-color: #ff3333; padding: 2px 6px; border-radius: 3px; color: #fff; font-weight: bold; font-size: 12px;">ショート戦略</span>
                <span style="margin-left: 5px;">（ここに具体的なエントリーポイント、利確目標、撤退ラインを記述。）</span>
            </li>
        </ul>
    </div>
    """
    
    # 動的モデル探索とフォールバック・カスケード（Canary Radar仕様）
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        preferred_order = [
            "models/gemini-1.5-flash-latest", "models/gemini-1.5-flash",
            "models/gemini-1.5-pro-latest", "models/gemini-1.5-pro", 
            "models/gemini-pro"
        ]
        
        target_model = None
        for pref in preferred_order:
            if pref in available_models:
                target_model = pref.replace("models/", "")
                break
                
        if not target_model and available_models:
            target_model = available_models[0].replace("models/", "")
            
        if not target_model:
            return "<p style='color: #ff4444;'>[エラー] 利用可能なGeminiモデルが見つかりません。</p>"

        # レートリミット回避のためのスリープ
        time.sleep(3)
        print(f"[*] Dynamic Model Discovery: AI Core '{target_model}' Engaged for {asset_name}.")
        model = genai.GenerativeModel(model_name=target_model)
        
        response = model.generate_content(prompt)
        out_text = response.text
        
        # AIが誤ってMarkdownのコードブロック記号を含めた場合のクレンジング
        out_text = re.sub(r'^```(?:html)?\s*', '', out_text)
        out_text = re.sub(r'\s*```$', '', out_text)
        
        return out_text.strip()
        
    except Exception as e:
        trace = traceback.format_exc()
        return f"<p style='color: #ff4444;'>[重大なエラー] AIインサイト生成に失敗しました (Model: {target_model})<br>詳細: {e}</p><!--\n{trace}\n-->"

def process_asset(asset_key, config):
    print(f"[*] Processing {config['name']}...")
    df_sb, df_gk, file_date, expiry_str = load_barchart_csv(asset_key)
    
    if df_sb is None or df_gk is None:
        print(f"[!] Data not found for {asset_key}. Skipping.")
        return None
        
    df = pd.merge(df_sb, df_gk, on='Strike', how='inner')
    
    # 週末データ欠落対策：過去5日分から最新値を取得
    try:
        ticker = yf.Ticker(config['ticker'])
        hist = ticker.history(period="5d")
        if hist.empty:
            raise ValueError("No price data found in 5d history")
        spot_price = float(hist['Close'].iloc[-1])
    except Exception as e:
        print(f"[!] Warning: yfinance failed to fetch {config['ticker']}. Using approx. Error: {e}")
        spot_price = df['Strike'].median()

    # Pandasマージ仕様のブレに強いカラム抽出
    df['Call_OpenInt'] = get_col_data(df, 'Open Int', 'Open Int_x').apply(clean_val)
    df['Put_OpenInt'] = get_col_data(df, 'Open Int.1', 'Open Int_y').apply(clean_val)
    df['Call_Gamma'] = get_col_data(df, 'Gamma', 'Gamma_x').apply(clean_val)
    df['Put_Gamma'] = get_col_data(df, 'Gamma.1', 'Gamma_y').apply(clean_val)
    df['Call_IV'] = get_col_data(df, 'IV', 'IV_x').apply(clean_val)
    df['Put_IV'] = get_col_data(df, 'IV.1', 'IV_y').apply(clean_val)
    
    df['Call_GEX'] = df['Call_Gamma'] * df['Call_OpenInt'] * 100 * spot_price * spot_price * 0.01 * config['multiplier']
    df['Put_GEX'] = df['Put_Gamma'] * df['Put_OpenInt'] * 100 * spot_price * spot_price * 0.01 * config['multiplier'] * -1
    
    df['Total_GEX'] = df['Call_GEX'] + df['Put_GEX']
    df['Call_GEX_M'] = df['Call_GEX'] / 1e6
    df['Put_GEX_M'] = df['Put_GEX'] / 1e6
    df['Total_GEX_M'] = df['Total_GEX'] / 1e6
    df['Avg_IV'] = (df['Call_IV'] + df['Put_IV']) / 2
    
    mask = (df['Call_GEX'] > 0) | (df['Put_GEX'] < 0)
    if not mask.any():
        return None
    valid_strikes = df.loc[mask, 'Strike']
    min_strike = valid_strikes.min()
    max_strike = valid_strikes.max()
    margin = (max_strike - min_strike) * 0.1
    df_filtered = df[(df['Strike'] >= min_strike - margin) & (df['Strike'] <= max_strike + margin)].copy()

    if df_filtered.empty:
        return None

    zero_gamma_idx = df_filtered['Total_GEX'].abs().idxmin()
    zero_gamma_strike = df_filtered.loc[zero_gamma_idx, 'Strike']
    call_wall_strike = df_filtered.loc[df_filtered['Call_GEX'].idxmax(), 'Strike']
    put_wall_strike = df_filtered.loc[df_filtered['Put_GEX'].idxmin(), 'Strike']

    if spot_price > zero_gamma_strike:
        regime = "🟢 POSITIVE GAMMA REGIME (押し目買い優位)"
        regime_color = "#00FF00"
    else:
        regime = "🔴 NEGATIVE GAMMA REGIME (パニック売り警戒)"
        regime_color = "#FF3333"

    print(f"[*] Generating AI Insight for {config['name']}...")
    ai_insight_html = generate_ai_insight(
        asset_name=config['name'],
        spot_price=spot_price,
        zero_gamma=zero_gamma_strike,
        call_wall=call_wall_strike,
        put_wall=put_wall_strike,
        regime=regime.split(" ")[0] + " " + regime.split(" ")[1]
    )

    # チャート描画
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.7, 0.3], vertical_spacing=0.05,
                        subplot_titles=(f"Dealer Net GEX Profile<br><span style='color:{regime_color}; font-size:16px;'>{regime}</span>", "Implied Volatility Smile"))

    fig.add_trace(go.Bar(
        x=df_filtered['Strike'], y=df_filtered['Call_GEX_M'],
        name="Call GEX (レジスタンス)", marker_color="#00FFFF", opacity=0.8
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df_filtered['Strike'], y=df_filtered['Put_GEX_M'],
        name="Put GEX (サポート)", marker_color="#FF00FF", opacity=0.8
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_filtered['Strike'], y=df_filtered['Total_GEX_M'],
        name="Net GEX", mode='lines+markers',
        line=dict(color='white', width=2), marker=dict(size=4)
    ), row=1, col=1)

    fig.add_vline(x=spot_price, line_width=2, line_dash="solid", line_color="yellow", row=1, col=1)
    fig.add_annotation(x=spot_price, y=-0.1, xref="x", yref="y domain", text=f"Current Spot<br>{spot_price}", showarrow=True, arrowhead=2, ax=0, ay=30, bgcolor="yellow", font=dict(color="black"), row=1, col=1)

    fig.add_vline(x=zero_gamma_strike, line_width=1.5, line_dash="dashdot", line_color="red", row=1, col=1)
    fig.add_annotation(x=zero_gamma_strike, y=0.95, xref="x", yref="y domain", text=f"Zero-Gamma<br>{zero_gamma_strike}", showarrow=True, arrowhead=2, ax=-40, ay=-30, bgcolor="red", font=dict(color="white"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_filtered['Strike'], y=df_filtered['Avg_IV'],
        name="IV", mode='lines+markers', line=dict(color='orange', width=2)
    ), row=2, col=1)

    fig.update_layout(
        title=f"Quant Options Radar: {config['name']} | Expiry: {expiry_str}<br><span style='font-size:12px;color:gray;'>As of: {file_date}</span>",
        template="plotly_dark", barmode='relative', hovermode="x unified", height=850,
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="GEX ($M)", row=1, col=1)
    fig.update_yaxes(title_text="IV (%)", row=2, col=1)
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)

    config_plotly = {'responsive': True, 'displayModeBar': False}
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config=config_plotly)

    return plot_html, ai_insight_html

def generate_html(asset_key, config, plot_html, ai_insight_html):
    tabs_html = ""
    for k, v in ASSET_CONFIG.items():
        active = "active" if k == asset_key else ""
        tabs_html += f'<a href="{v["filename"]}" class="tab {active}">{v["name"]}</a>'
    
    insight_panel = f"""
    <div class="ai-insight-panel">
        <h3 style="color: #ff9900; margin-top: 0; font-size: 16px; border-bottom: 1px solid #ff9900; padding-bottom: 8px;">
            <span style="font-size: 1.2em;">●</span> DAILY QUANT INSIGHT (Powered by Gemini AI)
        </h3>
        <div style="font-size: 14px; line-height: 1.6; color: #e0e0e0;">
            {ai_insight_html}
        </div>
    </div>
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Quant GEX Radar - {config['name']}</title>
        <style>
            body {{ background-color: #111; color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; }}
            .nav {{ background-color: #222; padding: 10px 20px; display: flex; flex-wrap: wrap; gap: 10px; border-bottom: 2px solid #333; }}
            .nav a {{ color: #aaa; text-decoration: none; padding: 8px 16px; border-radius: 4px; font-size: 14px; transition: 0.3s; background-color: #333; }}
            .nav a:hover {{ background-color: #444; color: white; }}
            .nav a.active {{ background-color: #0055ff; color: white; font-weight: bold; }}
            .container {{ padding: 20px; max-width: 1600px; margin: 0 auto; }}
            .chart-container {{ width: 100%; min-height: 850px; }}
            .ai-insight-panel {{
                background-color: #1a1a1a; border-left: 4px solid #ff9900; border-radius: 4px;
                padding: 20px; margin: 20px auto; max-width: 1500px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            .ai-insight-panel p {{ margin-bottom: 10px; }}
            .ai-insight-panel ul {{ margin-top: 0; padding-left: 20px; }}
            .ai-insight-panel li {{ margin-bottom: 5px; }}
            .ai-insight-panel strong {{ color: #ffffff; font-weight: 600; background-color: #333; padding: 2px 4px; border-radius: 3px; }}
            @media (max-width: 768px) {{
                .nav {{ padding: 10px; justify-content: center; }}
                .nav a {{ padding: 6px 12px; font-size: 12px; }}
                .container {{ padding: 10px; }}
                .chart-container {{ min-height: 600px; }}
                .ai-insight-panel {{ margin: 10px; padding: 15px; }}
            }}
        </style>
    </head>
    <body>
        <div class="nav">
            {tabs_html}
            <a href="gex_trading_guide.html" style="margin-left: auto; background-color: #ff9900; color: #111;">📖 取引マニュアル</a>
        </div>
        <div class="container">
            <div class="chart-container">
                {plot_html}
            </div>
            {insight_panel}
        </div>
    </body>
    </html>
    """
    
    filepath = DOCS_DIR / config['filename']
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    DOCS_DIR.mkdir(exist_ok=True)
    
    with open(DOCS_DIR / ".nojekyll", "w") as f:
        pass

    for asset_key, config in ASSET_CONFIG.items():
        try:
            result = process_asset(asset_key, config)
            if result:
                plot_html, ai_insight_html = result
                generate_html(asset_key, config, plot_html, ai_insight_html)
                print(f"[+] Successfully generated {config['filename']}")
            else:
                print(f"[-] Skipped {config['name']} due to missing or invalid data.")
        except Exception as e:
            print(f"[Error] Failed to process {config['name']}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
