import os
import glob
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
from pathlib import Path
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai

# ==========================================
# 設定
# ==========================================
ROOT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = ROOT_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

ASSET_CONFIG = {
    "ES": {"name": "S&P 500 (ES)", "ticker": "ES=F", "multiplier": 50, "filename": "es.html"},
    "SI": {"name": "シルバー (SI)", "ticker": "SI=F", "multiplier": 5000, "filename": "index.html"},
    "NG": {"name": "天然ガス (NG)", "ticker": "NG=F", "multiplier": 10000, "filename": "ng.html"},
    "HG": {"name": "銅 (HG)", "ticker": "HG=F", "multiplier": 25000, "filename": "hg.html"},
    "ZS": {"name": "大豆 (ZS)", "ticker": "ZS=F", "multiplier": 50, "filename": "zs.html"},
    "ZC": {"name": "トウモロコシ (ZC)", "ticker": "ZC=F", "multiplier": 50, "filename": "zc.html"},
    "ZW": {"name": "小麦 (ZW)", "ticker": "ZW=F", "multiplier": 50, "filename": "zw.html"}
}

# ==========================================
# ヘルパー関数
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
    # 先頭がティッカー名で始まり、特定のキーワードを含むファイルを検索
    sb_files = sorted(glob.glob(f"{prefix}*side-by-side*.csv"))
    gk_files = sorted(glob.glob(f"{prefix}*volatility-greeks*.csv"))
    
    if not sb_files or not gk_files:
        return None, None, None
        
    sb_path = sb_files[-1]
    gk_path = gk_files[-1]
    
    df_sb = pd.read_csv(sb_path)
    df_gk = pd.read_csv(gk_path)
    
    match = re.search(r'exp-(\d{2}_\d{2}_\d{2})', sb_path)
    expiry = match.group(1) if match else "Unknown"
    
    return df_sb, df_gk, expiry

# ==========================================
# AI インサイト生成 (w/ Smart Retry & Rate Limit Evasion)
# ==========================================
def generate_ai_insight(asset_name, spot, call_wall, put_wall, zero_gamma, regime):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "<p style='color:red;'>[エラー] APIキーが設定されていません。</p>"
        
    genai.configure(api_key=api_key)
    
    prompt = f"""
あなたは冷徹で論理的なクオンツ・アナリストです。以下のデータに基づき、指定したHTMLフォーマットにのみ従って作戦指令を出力してください。
Markdownのコードブロック（```html等）や、挨拶、追加の説明は絶対に含めないでください。デザインやHTML構造の改変は一切禁止します。

データ:
銘柄: {asset_name}
現在価格: {spot}
Call Wall (レジスタンス): {call_wall}
Put Wall (サポート): {put_wall}
Zero-Gamma: {zero_gamma}
レジーム: {regime}

出力フォーマット（必ず以下のHTML構造のまま、[ ] の部分だけを分析内容で埋めて出力すること）:
<div style="background:#1a1d21; padding:16px; border-radius:8px; border-left:4px solid var(--primary); font-family:sans-serif; line-height:1.6; margin-top:20px;">
    <h3 style="margin-top:0; color:#e0e0e0; font-size:16px; border-bottom:1px solid #333; padding-bottom:8px;">{asset_name} GEX オペレーション指令</h3>
    
    <div style="margin-bottom:12px;">
        <span style="background:rgba(20, 108, 46, 0.3); color:#44c265; padding:4px 8px; border-radius:4px; font-weight:bold; font-size:12px; margin-right:8px;">基本方針</span>
        <span style="color:#c4c7c5; font-size:14px;">[現状のレジームと価格位置に基づく基本方針を1文で記述]</span>
    </div>
    
    <div style="margin-bottom:12px;">
        <span style="background:rgba(11, 87, 208, 0.3); color:#76acff; padding:4px 8px; border-radius:4px; font-weight:bold; font-size:12px; margin-right:8px;">エントリー</span>
        <span style="color:#c4c7c5; font-size:14px;">[サポートからの反発やレジスタンス突破時のエントリー目安を簡潔に記述]</span>
    </div>
    
    <div>
        <span style="background:rgba(179, 38, 30, 0.3); color:#fe8983; padding:4px 8px; border-radius:4px; font-weight:bold; font-size:12px; margin-right:8px;">撤退ライン</span>
        <span style="color:#c4c7c5; font-size:14px;">[サポート割れ等の厳格な損切り・撤退ラインを簡潔に記述]</span>
    </div>
</div>
"""
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # 1日1500回の無料枠がある 1.5-flash を最優先にする (2.5-flashは1日20回制限で即死するため除外傾向)
        preferred_order = [
            "models/gemini-1.5-flash", "models/gemini-1.5-flash-latest", 
            "models/gemini-1.5-pro", "models/gemini-pro"
        ]
        target_models = [m for m in preferred_order if m in available_models]
        if not target_models:
            target_models = [available_models[0]] if available_models else ["gemini-1.5-flash"]
            
        last_error = "None"
        for m_name in target_models:
            model = genai.GenerativeModel(model_name=m_name.replace("models/", ""))
            
            for attempt in range(2):
                try:
                    response = model.generate_content(prompt)
                    html_res = response.text.replace('```html', '').replace('```', '').strip()
                    return html_res
                except Exception as e:
                    last_error = str(e)
                    if "429" in last_error or "Quota" in last_error:
                        print(f"[*] API Limit/Quota Hit on {m_name}. Waiting 60s... (Attempt {attempt+1})")
                        time.sleep(60)
                    else:
                        print(f"[-] Model {m_name} failed. Falling back.")
                        time.sleep(2)
                        break 
                        
        return f"<p style='color:red;'>[AI生成エラー] 制限超過または一時的な障害です。<br>詳細: {last_error}</p>"
        
    except Exception as e:
        return f"<p style='color:red;'>[致命的なエラー] API接続失敗: {e}</p>"

# ==========================================
# コア処理 (GEX算出パイプライン)
# ==========================================
def process_asset(asset_key, config):
    df_sb, df_gk, expiry = load_barchart_csv(asset_key)
    if df_sb is None:
        raise FileNotFoundError(f"CSV files not found for {asset_key}")
        
    # カラム名のクリーンアップ
    df_sb.columns = [str(c).strip() for c in df_sb.columns]
    df_gk.columns = [str(c).strip() for c in df_gk.columns]
    
    df_sb['Strike'] = df_sb['Strike'].apply(parse_strike)
    df_gk['Strike'] = df_gk['Strike'].apply(parse_strike)
    
    # 【最重要修正】横並び (Side-by-Side) CSVから、列名/インデックスで直接CallとPutを抜き出す
    # Side-by-Side ファイルからの OI 抽出
    call_oi_col = 'Open Int' if 'Open Int' in df_sb.columns else df_sb.columns[3]
    put_oi_col = 'Open Int.1' if 'Open Int.1' in df_sb.columns else df_sb.columns[9]
    df_sb['Call_OpenInt'] = df_sb[call_oi_col].apply(clean_val)
    df_sb['Put_OpenInt'] = df_sb[put_oi_col].apply(clean_val)
    
    # Greeks ファイルからの Gamma / IV 抽出
    gamma_call_col = 'Gamma' if 'Gamma' in df_gk.columns else df_gk.columns[3]
    gamma_put_col = 'Gamma.1' if 'Gamma.1' in df_gk.columns else df_gk.columns[13]
    iv_call_col = 'IV' if 'IV' in df_gk.columns else df_gk.columns[1]
    iv_put_col = 'IV.1' if 'IV.1' in df_gk.columns else df_gk.columns[11]
    
    df_gk['Gamma_Call'] = df_gk[gamma_call_col].apply(clean_val)
    df_gk['Gamma_Put'] = df_gk[gamma_put_col].apply(clean_val)
    df_gk['IV_Call'] = df_gk[iv_call_col].apply(clean_val)
    df_gk['IV_Put'] = df_gk[iv_put_col].apply(clean_val)

    # Strikeを軸にマージ
    df_merged = pd.merge(df_gk[['Strike', 'Gamma_Call', 'Gamma_Put', 'IV_Call', 'IV_Put']], 
                         df_sb[['Strike', 'Call_OpenInt', 'Put_OpenInt']], 
                         on='Strike', how='inner').fillna(0)
                         
    mult = config['multiplier']
    
    # Total GEX Calculation: (Gamma * OI * 100 * Multiplier) / 1,000,000
    df_merged['Call_GEX'] = df_merged['Gamma_Call'] * df_merged['Call_OpenInt'] * mult * 100 / 1e6 
    df_merged['Put_GEX'] = df_merged['Gamma_Put'] * df_merged['Put_OpenInt'] * mult * 100 * -1 / 1e6 
    df_merged['Total_GEX'] = df_merged['Call_GEX'] + df_merged['Put_GEX']
    
    # スポット価格の取得 (週末考慮 period="5d")
    spot_price = 0.0
    spot_date = datetime.now().strftime('%m-%d-%Y')
    try:
        hist = yf.Ticker(config['ticker']).history(period="5d")
        if not hist.empty:
            spot_price = float(hist['Close'].iloc[-1])
            spot_date = hist.index[-1].strftime('%m-%d-%Y')
    except Exception as e:
        print(f"Warning: Failed to fetch spot price for {config['ticker']}: {e}")
        
    if spot_price == 0.0:
        spot_price = df_merged['Strike'].median()

    # Zero-Gamma 算出 (流動性フィルタ ＋ 線形補間)
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

    call_wall_strike = df_filtered.loc[df_filtered['Call_GEX'].idxmax(), 'Strike']
    put_wall_strike = df_filtered.loc[df_filtered['Put_GEX'].idxmin(), 'Strike']
    
    regime = "POSITIVE GAMMA REGIME (押し目買い優位)" if spot_price > zero_gamma_strike else "NEGATIVE GAMMA REGIME (パニック売り警戒)"
    regime_color = "#44c265" if spot_price > zero_gamma_strike else "#fe8983"
    
    # ==========================================
    # グラフ描画
    # ==========================================
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Bar(x=df_filtered['Strike'], y=df_filtered['Call_GEX'], name='Call GEX (レジスタンス)', marker_color='#06bbdf'), row=1, col=1)
    fig.add_trace(go.Bar(x=df_filtered['Strike'], y=df_filtered['Put_GEX'], name='Put GEX (サポート)', marker_color='#c598ff'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_filtered['Strike'], y=df_filtered['Total_GEX'], mode='lines+markers', name='Net GEX', line=dict(color='white', width=2), marker=dict(size=4)), row=1, col=1)
    
    fig.add_vline(x=spot_price, line_width=2, line_dash="solid", line_color="yellow", row=1, col=1, annotation_text=f"Current Spot<br>{spot_price}", annotation_position="bottom right", annotation_bgcolor="yellow", annotation_font_color="black")
    fig.add_vline(x=zero_gamma_strike, line_width=1.5, line_dash="dashdot", line_color="red", row=1, col=1, annotation_text=f"Zero-Gamma<br>{zero_gamma_strike}", annotation_position="top left", annotation_bgcolor="red", annotation_font_color="white")
    
    fig.add_annotation(x=0.5, y=1.05, xref="paper", yref="paper", text=f"Dealer Net GEX Profile<br><span style='color:{regime_color}'>● {regime}</span>", showarrow=False, font=dict(size=14, color="white"), align="center")

    df_filtered['IV_Avg'] = (df_filtered['IV_Call'] + df_filtered['IV_Put']) / 2
    fig.add_trace(go.Scatter(x=df_filtered['Strike'], y=df_filtered['IV_Avg'], mode='lines+markers', name='IV', line=dict(color='orange', width=2)), row=2, col=1)
    
    fig.update_layout(
        title=f"Quant Options Radar: {config['name']} | Expiry: {expiry}<br><sup style='font-size:12px;color:#c4c7c5'>As of: {spot_date}</sup>",
        template="plotly_dark", paper_bgcolor="#101218", plot_bgcolor="#101218",
        barmode='overlay', hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="GEX ($M)", row=1, col=1, gridcolor="#2d2f38")
    fig.update_yaxes(title_text="IV (%)", row=2, col=1, gridcolor="#2d2f38")
    fig.update_xaxes(title_text="Strike Price", row=2, col=1, gridcolor="#2d2f38")
    fig.update_xaxes(gridcolor="#2d2f38", row=1, col=1)

    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # AIインサイト呼び出し
    ai_html = generate_ai_insight(config['name'], spot_price, call_wall_strike, put_wall_strike, zero_gamma_strike, regime)

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
            .ai-header {{ color: #f9ab00; font-weight: bold; font-size: 14px; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }}
        </style>
    </head>
    <body>
        <div class="nav-tabs">
            {''.join([f'<a href="{cfg["filename"]}" class="{"active" if k == asset_key else ""}">{cfg["name"]}</a>' for k, cfg in ASSET_CONFIG.items()])}
            <a href="gex_trading_guide.html" style="margin-left:auto; color: #f9ab00;">■ 取引マニュアル</a>
        </div>
        <div class="container">
            {graph_html}
            <div class="ai-panel">
                <div class="ai-header">● DAILY QUANT INSIGHT (Powered by Gemini AI)</div>
                {ai_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(DOCS_DIR / config['filename'], "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"[+] Successfully generated {config['filename']}")

def main():
    assets = list(ASSET_CONFIG.items())
    for i, (key, config) in enumerate(assets):
        print(f"\n[*] Processing {config['name']}...")
        try:
            process_asset(key, config)
            # レートリミット回避のためのインターバル
            if i < len(assets) - 1:
                print(f"[...] Waiting 10 seconds before next asset to avoid Rate Limits.")
                time.sleep(10) 
        except Exception as e:
            print(f"Error: Failed to process {config['name']}: {e}")

if __name__ == "__main__":
    main()
