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
    "SI": {"name": "🥈 シルバー (SI)", "ticker": "SI=F", "multiplier": 5000, "filename": "index.html"},
    "NG": {"name": "🔥 天然ガス (NG)", "ticker": "NG=F", "multiplier": 10000, "filename": "ng.html"},
    "HG": {"name": "🧱 銅 (HG)", "ticker": "HG=F", "multiplier": 25000, "filename": "hg.html"},
    "ZS": {"name": "🌱 大豆 (ZS)", "ticker": "ZS=F", "multiplier": 50, "filename": "zs.html"},
    "ZC": {"name": "🌽 トウモロコシ (ZC)", "ticker": "ZC=F", "multiplier": 50, "filename": "zc.html"},
    "ZW": {"name": "🌾 小麦 (ZW)", "ticker": "ZW=F", "multiplier": 50, "filename": "zw.html"},
    "SB": {"name": "🍬 砂糖 (SB)", "ticker": "SB=F", "multiplier": 1120, "filename": "sb.html"}
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
    
    # ファイル名から「データ取得日（As of）」を正確に抽出するロジックを追加
    date_match = re.search(r'-(\d{2}-\d{2}-\d{4})\.csv', sb_path)
    as_of_date = date_match.group(1) if date_match else None
    
    return df_sb, df_gk, expiry, as_of_date

def generate_batched_insights(asset_summaries):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[-] Error: GEMINI_API_KEY is missing.")
        return {k: "<p style='color:#fe8983;'>[エラー] APIキーが設定されていません。</p>" for k in asset_summaries.keys()}
        
    genai.configure(api_key=api_key)
    
    # バッチ処理により1回の実行につき1リクエストで済むため、
    # 最も賢い最新の 2.5-flash（1日20回制限）をメインエンジンに復帰。
    # APIの仕様変更に備えたフォールバック・アレイを定義。
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest"
    ]
    
    # 全アセットのデータを含んだプロンプトを構築
    prompt = f"""
あなたは金融工学とオプション取引に精通した「リード・クオンツアナリスト」です。
以下の全銘柄の最新のGEX（ガンマ・エクスポージャー）データに基づき、オプション初心者にも分かるように現在の重力場の分析と、実践的なトレード戦略をJSONフォーマットで出力してください。

【全銘柄のデータ】
{json.dumps(asset_summaries, ensure_ascii=False, indent=2)}

【各銘柄のインサイト出力ルール (厳守事項)】
- Webページに直接埋め込むため、各銘柄の値(Value)には純粋なHTMLの断片のみを文字列として出力すること。
- 以下のHTMLタグを駆使して構造化すること: <h3>, <ul>, <li>, <p>, <strong>
- ダークテーマのダッシュボードに映えるよう、重要な数値や方向性にはインラインCSSで色付けをすること。（HTMLの属性にはシングルクォート「'」を使用し、JSONを壊さないこと。例: <span style='color: #44c265;'>）
- 各銘柄のデータに含まれる「Call Wall（レジスタンス）」「Put Wall（サポート）」の数値を具体的に引用しながら、現在のレジームに基づく「どこでエントリーし、どこで利確・損切りすべきか」の具体的なアクションプランにフォーカスすること。

【出力すべきJSONフォーマット】
Markdownのコードブロック記号(```jsonなど)や挨拶は一切不要です。純粋なJSONオブジェクトのみを出力してください。
{{
  "ES": "<h3>🇺🇸 S&P 500 (ES) 分析</h3><ul><li>...</li></ul>",
  "SI": "<h3>🥈 シルバー (SI) 分析</h3><ul><li>...</li></ul>",
  ...
}}
"""
    
    last_error = None
    for target_model in models_to_try:
        print(f"[*] Trying AI Model: {target_model}...")
        try:
            model = genai.GenerativeModel(model_name=target_model)
            response = model.generate_content(prompt)
            res_text = response.text.strip()
            
            # 正規表現でJSON部分のみを抽出（AIの無駄な挨拶やMarkdownを完璧にパージ）
            match = re.search(r'\{.*\}', res_text, re.DOTALL)
            if match:
                res_text = match.group(0)
                
            insights = json.loads(res_text)
            print(f"[+] AI Insights successfully generated via {target_model}.")
            return insights
        except Exception as e:
            print(f"[-] Model {target_model} failed: {e}")
            last_error = e
            
    print("[-] All AI models failed in the fallback array.")
    err_msg = f"<p style='color:#fe8983;'>[AI生成エラー] 制限超過またはAPIの仕様変更による一時的な障害です。<br>詳細: {last_error}</p>"
    return {k: err_msg for k in asset_summaries.keys()}

def process_asset_data(asset_key, config):
    df_sb, df_gk, expiry, file_as_of = load_barchart_csv(asset_key)
    if df_sb is None:
        raise FileNotFoundError(f"CSV files not found for {asset_key}")
        
    df_sb.columns = [str(c).strip() for c in df_sb.columns]
    df_gk.columns = [str(c).strip() for c in df_gk.columns]
    
    df_sb['Strike'] = df_sb['Strike'].apply(parse_strike)
    df_gk['Strike'] = df_gk['Strike'].apply(parse_strike)
    
    # 堅牢なインデックスベースの列抽出
    oi_idx = [i for i, col in enumerate(df_sb.columns) if 'Open Int' in col or 'OI' in col]
    if len(oi_idx) >= 2:
        df_sb['Call_OpenInt'] = df_sb.iloc[:, oi_idx[0]].apply(clean_val)
        df_sb['Put_OpenInt'] = df_sb.iloc[:, oi_idx[1]].apply(clean_val)
    else:
        df_sb['Call_OpenInt'] = 0.0
        df_sb['Put_OpenInt'] = 0.0

    gamma_idx = [i for i, col in enumerate(df_gk.columns) if 'Gamma' in col]
    iv_idx = [i for i, col in enumerate(df_gk.columns) if 'IV' in col and 'Skew' not in col]

    if len(gamma_idx) >= 2:
        df_gk['Gamma_Call'] = df_gk.iloc[:, gamma_idx[0]].apply(clean_val)
        df_gk['Gamma_Put'] = df_gk.iloc[:, gamma_idx[1]].apply(clean_val)
    else:
        df_gk['Gamma_Call'] = 0.0
        df_gk['Gamma_Put'] = 0.0

    if len(iv_idx) >= 2:
        df_gk['IV_Call'] = df_gk.iloc[:, iv_idx[0]].apply(clean_val)
        df_gk['IV_Put'] = df_gk.iloc[:, iv_idx[1]].apply(clean_val)
    else:
        df_gk['IV_Call'] = 0.0
        df_gk['IV_Put'] = 0.0

    df_sb_agg = df_sb.groupby('Strike', as_index=False)[['Call_OpenInt', 'Put_OpenInt']].sum()
    df_gk_agg = df_gk.groupby('Strike', as_index=False)[['Gamma_Call', 'Gamma_Put', 'IV_Call', 'IV_Put']].max()

    df_merged = df_gk_agg.merge(df_sb_agg, on='Strike', how='outer').fillna(0)
                         
    mult = config['multiplier']
    df_merged['Call_GEX'] = df_merged['Gamma_Call'] * df_merged['Call_OpenInt'] * mult * 100 / 1e6 
    df_merged['Put_GEX'] = df_merged['Gamma_Put'] * df_merged['Put_OpenInt'] * mult * 100 * -1 / 1e6 
    df_merged['Total_GEX'] = df_merged['Call_GEX'] + df_merged['Put_GEX']
    
    spot_price = 0.0
    yf_date = datetime.now().strftime('%m-%d-%Y')
    try:
        hist = yf.Ticker(config['ticker']).history(period="5d")
        if not hist.empty:
            spot_price = float(hist['Close'].iloc[-1])
            yf_date = hist.index[-1].strftime('%m-%d-%Y')
    except Exception as e:
        print(f"Warning: Failed to fetch spot price for {config['ticker']}: {e}")
        
    if spot_price == 0.0:
        spot_price = df_merged['Strike'].median()

    # CSVファイル名から抽出した日付を最優先とし、データの鮮度を正確に反映
    spot_date = file_as_of if file_as_of else yf_date

    min_strike = spot_price * 0.8
    max_strike = spot_price * 1.2
    margin = (max_strike - min_strike) * 0.1
    df_filtered = df_merged[(df_merged['Strike'] >= min_strike - margin) & (df_merged['Strike'] <= max_strike + margin)].copy()

    if df_filtered.empty:
        df_filtered = df_merged.copy()

    df_sorted = df_filtered.sort_values('Strike').reset_index(drop=True)
    df_sorted['Total_OI'] = df_sorted['Call_OpenInt'] + df_sorted['Put_OpenInt']
    valid_mask = df_sorted['Total_OI'] > df_sorted['Total_OI'].max() * 0.05 
    df_valid = df_sorted[valid_mask].reset_index(drop=True)
    
    if not df_valid.empty:
        signs = np.sign(df_valid['Total_GEX'])
        flips = np.where(np.diff(signs) != 0)[0] 
        
        if len(flips) > 0:
            closest_flip_strike = None
            min_dist = float('inf')
            for idx in flips:
                s1, s2 = df_valid.loc[idx, 'Strike'], df_valid.loc[idx + 1, 'Strike']
                g1, g2 = df_valid.loc[idx, 'Total_GEX'], df_valid.loc[idx + 1, 'Total_GEX']
                exact_zero_strike = s1 - g1 * (s2 - s1) / (g2 - g1) if g1 != g2 else (s1 + s2) / 2.0
                dist = abs(exact_zero_strike - spot_price)
                if dist < min_dist:
                    min_dist = dist
                    closest_flip_strike = exact_zero_strike
            zero_gamma_strike = round(closest_flip_strike, 2)
        else:
            zero_gamma_idx = df_valid['Total_GEX'].abs().idxmin()
            zero_gamma_strike = df_valid.loc[zero_gamma_idx, 'Strike']
    else:
        zero_gamma_strike = spot_price

    call_walls_df = df_filtered[df_filtered['Call_GEX'] > 0].nlargest(2, 'Call_GEX')[['Strike', 'Call_GEX']]
    put_walls_df = df_filtered[df_filtered['Put_GEX'] < 0].nsmallest(2, 'Put_GEX')[['Strike', 'Put_GEX']]
    
    call_walls = call_walls_df.to_dict('records')
    put_walls = put_walls_df.to_dict('records')
    
    while len(call_walls) < 2: call_walls.append({"Strike": 0.0, "Call_GEX": 0.0})
    while len(put_walls) < 2: put_walls.append({"Strike": 0.0, "Put_GEX": 0.0})
    
    is_positive = spot_price > zero_gamma_strike
    regime_str = "POSITIVE GAMMA REGIME (押し目買い優位)" if is_positive else "NEGATIVE GAMMA REGIME (パニック売り警戒)"
    regime_color = "#44c265" if is_positive else "#fe8983"
    
    data_summary = {
        "asset_name": config['name'],
        "spot": round(spot_price, 3),
        "zero_gamma": zero_gamma_strike,
        "regime": regime_str,
        "call_walls": call_walls,
        "put_walls": put_walls
    }
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Bar(x=df_filtered['Strike'], y=df_filtered['Call_GEX'], name='Call GEX (レジスタンス)', marker_color='#06bbdf'), row=1, col=1)
    fig.add_trace(go.Bar(x=df_filtered['Strike'], y=df_filtered['Put_GEX'], name='Put GEX (サポート)', marker_color='#c598ff'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_filtered['Strike'], y=df_filtered['Total_GEX'], mode='lines+markers', name='Net GEX', line=dict(color='white', width=2), marker=dict(size=4)), row=1, col=1)
    
    fig.add_vline(x=spot_price, line_width=2, line_dash="solid", line_color="yellow", row=1, col=1, annotation_text=f"Current Spot<br>{spot_price:.3f}", annotation_position="bottom right", annotation_bgcolor="yellow", annotation_font_color="black")
    fig.add_vline(x=zero_gamma_strike, line_width=1.5, line_dash="dashdot", line_color="red", row=1, col=1, annotation_text=f"Zero-Gamma<br>{zero_gamma_strike:.2f}", annotation_position="top left", annotation_bgcolor="red", annotation_font_color="white")
    
    df_filtered['IV_Avg'] = (df_filtered['IV_Call'] + df_filtered['IV_Put']) / 2
    fig.add_trace(go.Scatter(x=df_filtered['Strike'], y=df_filtered['IV_Avg'], mode='lines+markers', name='IV', line=dict(color='orange', width=2)), row=2, col=1)
    
    # タイトル部分にレジーム表記を統合し、グラフ内のテキスト重なりを解消
    fig.update_layout(
        title=f"Quant Options Radar: {config['name']} | Expiry: {expiry}<br><sup style='font-size:12px;color:#c4c7c5'>As of: {spot_date}</sup><br><br><span style='font-size:16px;color:{regime_color}'>● {regime_str}</span>",
        template="plotly_dark", paper_bgcolor="#101218", plot_bgcolor="#101218",
        barmode='overlay', hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=120)
    )
    fig.update_yaxes(title_text="GEX ($M)", row=1, col=1, gridcolor="#2d2f38")
    fig.update_yaxes(title_text="IV (%)", row=2, col=1, gridcolor="#2d2f38")
    fig.update_xaxes(title_text="Strike Price", row=2, col=1, gridcolor="#2d2f38")
    fig.update_xaxes(gridcolor="#2d2f38", row=1, col=1)

    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    return graph_html, data_summary, expiry

def main():
    graphs = {}
    asset_summaries = {}
    
    for key, config in ASSET_CONFIG.items():
        print(f"[*] Processing {config['name']}...")
        try:
            graph_html, summary, _ = process_asset_data(key, config)
            graphs[key] = graph_html
            asset_summaries[key] = summary
        except Exception as e:
            print(f"[-] Error: Failed to process {config['name']}: {e}")
            graphs[key] = f"<p style='color:red;'>データ処理エラー: {e}</p>"

    print("\n[*] Dispatching batch request to Gemini AI...")
    ai_insights = {}
    if asset_summaries:
        ai_insights = generate_batched_insights(asset_summaries)

    for key, config in ASSET_CONFIG.items():
        print(f"[*] Building HTML for {config['name']}...")
        graph_html = graphs.get(key, "")
        
        insight_content = ai_insights.get(key, "<p style='color:#fe8983;'>インサイトデータの取得に失敗しました。</p>")
        if isinstance(insight_content, dict):
            insight_content = insight_content.get("error", str(insight_content))

        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja" data-theme="dark">
        <head>
            <meta charset="UTF-8">
            <title>Quant GEX Radar - {config['name']}</title>
            <style>
                body {{ background-color: #101218; color: #ffffff; font-family: sans-serif; margin: 0; padding: 0; }}
                .nav-tabs {{ background: #1a1d21; padding: 10px; display: flex; gap: 10px; overflow-x: auto; }}
                .nav-tabs a {{ color: #c4c7c5; text-decoration: none; padding: 8px 16px; border-radius: 4px; font-size: 14px; white-space: nowrap; }}
                .nav-tabs a:hover {{ background: #2d2f38; }}
                .nav-tabs a.active {{ background: #0b57d0; color: white; font-weight: bold; }}
                .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
                .ai-panel {{ margin-top: 30px; border-top: 1px solid #2d2f38; padding-top: 20px; }}
                .ai-header {{ color: #f9ab00; font-weight: bold; font-size: 14px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px; }}
                .ai-content {{ background: #1a1d21; padding: 20px; border-radius: 8px; border-left: 4px solid var(--primary); font-size: 14px; line-height: 1.6; color: #c4c7c5; }}
                .ai-content h3 {{ color: #e0e0e0; font-size: 16px; border-bottom: 1px solid #333; padding-bottom: 8px; margin-top: 0; }}
                .ai-content ul {{ padding-left: 20px; }}
                .ai-content li {{ margin-bottom: 8px; }}
            </style>
        </head>
        <body>
            <div class="nav-tabs">
                {''.join([f'<a href="{cfg["filename"]}" class="{"active" if k == key else ""}">{cfg["name"]}</a>' for k, cfg in ASSET_CONFIG.items()])}
                <a href="gex_trading_guide.html" style="margin-left:auto; color: #f9ab00;">■ 取引マニュアル</a>
            </div>
            <div class="container">
                {graph_html}
                <div class="ai-panel">
                    <div class="ai-header">● DAILY QUANT INSIGHT (Powered by Gemini AI)</div>
                    <div class="ai-content">
                        {insight_content}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(DOCS_DIR / config['filename'], "w", encoding="utf-8") as f:
            f.write(html_content)
            
        print(f"[+] Successfully saved {config['filename']}")

if __name__ == "__main__":
    main()
