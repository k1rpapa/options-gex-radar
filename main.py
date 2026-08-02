import os
import sys
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

# Windowsコンソールでの文字化け・絵文字出力エラー防止
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

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
    "HE": {"name": "🐷 豚肉 (HE)", "ticker": "HE=F", "multiplier": 400, "filename": "he.html"},
    "CC": {"name": "🍫 ココア (CC)", "ticker": "CC=F", "multiplier": 10, "filename": "cc.html"},
    "KC": {"name": "☕ コーヒー (KC)", "ticker": "KC=F", "multiplier": 375, "filename": "kc.html"}
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
    
    match = re.search(r'exp-(\d{2}_\d{2}_\d{2})', sb_path)
    expiry = match.group(1) if match else "Unknown"
    
    date_match = re.search(r'-(\d{2}-\d{2}-\d{4})\.csv', sb_path)
    as_of_date = date_match.group(1) if date_match else None
    
    return df_sb, df_gk, expiry, as_of_date

def generate_batched_insights(asset_summaries):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[-] Error: GEMINI_API_KEY is missing.")
        return {k: "<p style='color:#fe8983;'>[エラー] APIキーが設定されていません。</p>" for k in asset_summaries.keys()}
        
    genai.configure(api_key=api_key)
    
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest"
    ]
    
    # シンタックスハイライトを壊さないための安全な文字列結合
    prompt = (
        "あなたは金融工学とオプション取引に精通した「リード・クオンツアナリスト」です。\n"
        "以下の全銘柄の最新のGEX（ガンマ・エクスポージャー）データに基づき、オプション初心者にも分かるように現在の重力場の分析と、実践的なトレード戦略をJSONフォーマットで出力してください。\n\n"
        "【全銘柄のデータ】\n"
        f"{json.dumps(asset_summaries, ensure_ascii=False, indent=2)}\n\n"
        "【各銘柄のインサイト出力ルール (厳守事項)】\n"
        "- Webページに直接埋め込むため、各銘柄の値(Value)には純粋なHTMLの断片のみを文字列として出力すること。\n"
        "- 以下のHTMLタグを駆使して構造化すること: <h3>, <ul>, <li>, <p>, <strong>\n"
        "- ダークテーマのダッシュボードに映えるよう、重要な数値や方向性にはインラインCSSで色付けをすること。（HTMLの属性にはシングルクォート「'」を使用し、JSONを壊さないこと。例: <span style='color: #44c265;'>）\n"
        "- 各銘柄のデータに含まれる「Call Wall（レジスタンス）」「Put Wall（サポート）」の数値を具体的に引用しながら、現在のレジームに基づく「どこでエントリーし、どこで利確・損切りすべきか」の具体的なアクションプランにフォーカスすること。\n\n"
        "【出力すべきJSONフォーマット】\n"
        "Markdownのコードブロック記号(バッククォート3つなど)や挨拶は一切不要です。純粋なJSONオブジェクトのみを出力してください。\n"
        "{\n"
        '  "ES": "<h3>🇺🇸 S&P 500 (ES) 分析</h3><ul><li>...</li></ul>",\n'
        '  "SI": "<h3>🥈 シルバー (SI) 分析</h3><ul><li>...</li></ul>",\n'
        '  ...\n'
        "}"
    )
    
    last_error = None
    for target_model in models_to_try:
        print(f"[*] Trying AI Model: {target_model}...")
        try:
            model = genai.GenerativeModel(model_name=target_model)
            response = model.generate_content(prompt)
            res_text = response.text.strip()
            
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
    try:
        hist = yf.Ticker(config['ticker']).history(period="5d")
        if not hist.empty:
            spot_price = float(hist['Close'].iloc[-1])
    except Exception as e:
        print(f"Warning: Failed to fetch spot price for {config['ticker']}: {e}")
        
    if spot_price == 0.0:
        spot_price = df_merged['Strike'].median()

    spot_date = file_as_of if file_as_of else datetime.now().strftime('%m-%d-%Y')

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
    
    fig = make_subplots(
        rows=1, cols=2, 
        shared_yaxes=True, 
        horizontal_spacing=0.03, 
        column_widths=[0.72, 0.28],
        subplot_titles=("GEX Option Ladder ($M)", "IV Profile (%)")
    )
    
    fig.add_trace(go.Bar(y=df_filtered['Strike'], x=df_filtered['Put_GEX'], orientation='h', name='Put GEX (サポート)', marker_color='#c598ff'), row=1, col=1)
    fig.add_trace(go.Bar(y=df_filtered['Strike'], x=df_filtered['Call_GEX'], orientation='h', name='Call GEX (レジスタンス)', marker_color='#06bbdf'), row=1, col=1)
    fig.add_trace(go.Scatter(y=df_filtered['Strike'], x=df_filtered['Total_GEX'], mode='lines+markers', name='Net GEX', line=dict(color='white', width=2), marker=dict(size=4)), row=1, col=1)
    
    fig.add_hline(y=spot_price, line_width=2, line_dash="solid", line_color="yellow", row=1, col=1, annotation_text=f"Current Spot: {spot_price:.3f}", annotation_position="top right", annotation_bgcolor="yellow", annotation_font_color="black")
    fig.add_hline(y=zero_gamma_strike, line_width=1.5, line_dash="dashdot", line_color="red", row=1, col=1, annotation_text=f"Zero-Gamma: {zero_gamma_strike:.2f}", annotation_position="bottom left", annotation_bgcolor="red", annotation_font_color="white")
    
    df_filtered['IV_Avg'] = (df_filtered['IV_Call'] + df_filtered['IV_Put']) / 2
    fig.add_trace(go.Scatter(y=df_filtered['Strike'], x=df_filtered['IV_Avg'], mode='lines+markers', name='IV', line=dict(color='orange', width=2)), row=1, col=2)
    
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#101218", plot_bgcolor="#101218",
        barmode='overlay', hovermode="y unified",
        showlegend=False,
        margin=dict(t=25, b=40, l=60, r=40),
        height=680
    )
    fig.update_yaxes(title_text="Strike Price", row=1, col=1, gridcolor="#2d2f38")
    fig.update_xaxes(title_text="GEX ($M)", row=1, col=1, gridcolor="#2d2f38")
    fig.update_xaxes(title_text="IV (%)", row=1, col=2, gridcolor="#2d2f38")

    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
    
    header_info = {
        "expiry": expiry,
        "spot_date": spot_date,
        "regime_str": regime_str,
        "regime_color": regime_color,
        "is_positive": is_positive
    }
    return graph_html, data_summary, header_info

def main():
    graphs = {}
    asset_summaries = {}
    headers_info = {}
    
    for key, config in ASSET_CONFIG.items():
        print(f"[*] Processing {config['name']}...")
        try:
            graph_html, summary, header_info = process_asset_data(key, config)
            graphs[key] = graph_html
            asset_summaries[key] = summary
            headers_info[key] = header_info
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
        info = headers_info.get(key, {
            "expiry": "N/A", "spot_date": "N/A", 
            "regime_str": "UNKNOWN", "regime_color": "#ffffff", "is_positive": True
        })
        
        insight_content = ai_insights.get(key, "<p style='color:#fe8983;'>インサイトデータの取得に失敗しました。</p>")
        if isinstance(insight_content, dict):
            insight_content = insight_content.get("error", str(insight_content))

        tabs_links = []
        for k, cfg in ASSET_CONFIG.items():
            active_cls = "active" if k == key else ""
            tabs_links.append(f'<a href="{cfg["filename"]}" class="{active_cls}">{cfg["name"]}</a>')
        tabs_html = "\n                ".join(tabs_links)

        regime_bg = "rgba(68, 194, 101, 0.15)" if info['is_positive'] else "rgba(254, 137, 131, 0.15)"

        # シンタックスハイライトを壊さないための安全な文字列結合
        html_content = (
            '<!DOCTYPE html>\n'
            '<html lang="ja" data-theme="dark">\n'
            '<head>\n'
            '    <meta charset="UTF-8">\n'
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'    <title>Quant GEX Radar - {config["name"]}</title>\n'
            '    <style>\n'
            '        * { box-sizing: border-box; }\n'
            '        body { background-color: #101218; color: #ffffff; font-family: sans-serif; margin: 0; padding: 0; }\n'
            '        .nav-tabs { background: #1a1d21; padding: 10px; display: flex; gap: 10px; overflow-x: auto; position: sticky; top: 0; z-index: 100; }\n'
            '        .nav-tabs a { color: #c4c7c5; text-decoration: none; padding: 8px 16px; border-radius: 4px; font-size: 14px; white-space: nowrap; }\n'
            '        .nav-tabs a:hover { background: #2d2f38; }\n'
            '        .nav-tabs a.active { background: #0b57d0; color: white; font-weight: bold; }\n'
            '        .container { max-width: 1800px; margin: 0 auto; padding: 15px; width: 100%; }\n'
            '        .dashboard-grid { display: flex; flex-direction: column; gap: 20px; width: 100%; }\n'
            '        .chart-panel { width: 100%; min-width: 0; }\n'
            '        .ai-panel { width: 100%; min-width: 0; border-top: 1px solid #2d2f38; padding-top: 15px; }\n'
            '        .panel-header { margin-bottom: 12px; display: flex; flex-direction: column; gap: 8px; }\n'
            '        .chart-main-title { font-size: 18px; font-weight: bold; color: #ffffff; line-height: 1.3; }\n'
            '        .chart-sub-info { font-size: 12px; color: #c4c7c5; }\n'
            '        .regime-badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: bold; align-self: flex-start; }\n'
            '        .html-legend { display: flex; flex-wrap: wrap; gap: 15px; font-size: 12px; color: #c4c7c5; margin-top: 4px; }\n'
            '        .legend-item { display: flex; align-items: center; gap: 6px; }\n'
            '        .color-dot { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }\n'
            '        @media (min-width: 992px) {\n'
            '            .dashboard-grid { flex-direction: row; align-items: flex-start; justify-content: space-between; }\n'
            '            .chart-panel { width: 55%; flex: 0 0 55%; min-width: 0; position: sticky; top: 60px; }\n'
            '            .ai-panel { width: 43%; flex: 0 0 43%; min-width: 0; border-top: none; padding-top: 0; }\n'
            '        }\n'
            '        .ai-header { color: #f9ab00; font-weight: bold; font-size: 16px; display: flex; align-items: center; gap: 8px; line-height: 1.3; margin-bottom: 0; }\n'
            '        .ai-content { background: #1a1d21; padding: 20px; border-radius: 8px; border-left: 4px solid #0b57d0; font-size: 14px; line-height: 1.6; color: #c4c7c5; margin-top: 8px; }\n'
            '        .ai-content h3 { color: #e0e0e0; font-size: 16px; border-bottom: 1px solid #333; padding-bottom: 8px; margin-top: 0; }\n'
            '        .ai-content ul { padding-left: 20px; }\n'
            '        .ai-content li { margin-bottom: 8px; }\n'
            '    </style>\n'
            '</head>\n'
            '<body>\n'
            '    <div class="nav-tabs">\n'
            f'        {tabs_html}\n'
            '        <a href="gex_trading_guide.html" style="margin-left:auto; color: #f9ab00;">■ 取引マニュアル</a>\n'
            '    </div>\n'
            '    <div class="container">\n'
            '        <div class="dashboard-grid">\n'
            '            <div class="chart-panel">\n'
            '                <div class="panel-header">\n'
            f'                    <div class="chart-main-title">Quant Options Radar: {config["name"]}</div>\n'
            f'                    <div class="chart-sub-info">Expiry: {info["expiry"]} &nbsp;|&nbsp; As of: {info["spot_date"]}</div>\n'
            f'                    <div class="regime-badge" style="background:{regime_bg}; color:{info["regime_color"]}; border:1px solid {info["regime_color"]};">● {info["regime_str"]}</div>\n'
            '                    <div class="html-legend">\n'
            '                        <span class="legend-item"><span class="color-dot" style="background:#c598ff"></span>Put GEX (サポート)</span>\n'
            '                        <span class="legend-item"><span class="color-dot" style="background:#06bbdf"></span>Call GEX (レジスタンス)</span>\n'
            '                        <span class="legend-item"><span class="color-dot" style="background:#ffffff"></span>Net GEX</span>\n'
            '                        <span class="legend-item"><span class="color-dot" style="background:#ffa500"></span>IV</span>\n'
            '                    </div>\n'
            '                </div>\n'
            f'                {graph_html}\n'
            '            </div>\n'
            '            <div class="ai-panel">\n'
            '                <div class="panel-header">\n'
            '                    <div class="ai-header">● DAILY QUANT INSIGHT (Powered by Gemini AI)</div>\n'
            '                </div>\n'
            '                <div class="ai-content">\n'
            f'                    {insight_content}\n'
            '                </div>\n'
            '            </div>\n'
            '        </div>\n'
            '    </div>\n'
            '</body>\n'
            '</html>'
        )
        
        with open(DOCS_DIR / config['filename'], "w", encoding="utf-8") as f:
            f.write(html_content)
            
        print(f"[+] Successfully saved {config['filename']}")

if __name__ == "__main__":
    main()
