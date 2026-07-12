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
import google.generativeai as genai  # ← 【追加】

# パス定義
ROOT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = ROOT_DIR / "docs"

# 複数アセット定義
ASSET_CONFIG = {
    "SI": {"name": "銀先物 (SI)", "ticker": "SI=F", "multiplier": 5000, "filename": "index.html"},
    "NG": {"name": "天然ガス先物 (NG)", "ticker": "NG=F", "multiplier": 10000, "filename": "ng.html"},
    "HG": {"name": "銅先物 (HG)", "ticker": "HG=F", "multiplier": 25000, "filename": "hg.html"},
    "ZS": {"name": "大豆先物 (ZS)", "ticker": "ZS=F", "multiplier": 50, "filename": "zs.html"},
    "ZC": {"name": "コーン先物 (ZC)", "ticker": "ZC=F", "multiplier": 50, "filename": "zc.html"},
    "ZW": {"name": "小麦先物 (ZW)", "ticker": "ZW=F", "multiplier": 50, "filename": "zw.html"}
}

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
    
    df_sb = pd.read_csv(sb_path)
    df_gk = pd.read_csv(gk_path)
    df_sb.columns = [str(c).strip() for c in df_sb.columns]
    df_gk.columns = [str(c).strip() for c in df_gk.columns]
    
    df_sb['Strike'] = df_sb['Strike'].apply(parse_strike)
    df_gk['Strike'] = df_gk['Strike'].apply(parse_strike)

    oi_candidates = ['Open Int', 'OI', 'Open Interest']
    call_oi_col, put_oi_col = None, None
    for col in oi_candidates:
        if col in df_sb.columns:
            call_oi_col, put_oi_col = col, f"{col}.1"
            break

    iv_candidates = ['IV', 'Implied Vol', 'Implied Volatility']
    call_iv_col, put_iv_col = None, None
    for col in iv_candidates:
        if col in df_gk.columns:
            call_iv_col, put_iv_col = col, f"{col}.1"
            break
            
    if not call_oi_col or not call_iv_col:
        raise KeyError("必要なカラムが見つかりません。")
        
    df = pd.merge(df_sb[['Strike', call_oi_col, put_oi_col]], 
                  df_gk[['Strike', call_iv_col, put_iv_col]], 
                  on='Strike', how='inner')
    
    def clean_val(x):
        s = str(x).split('-')[0].replace('%', '').replace(',', '').replace('N/A', '0').replace('nan', '0').strip()
        try: return float(s)
        except: return 0.0

    df['Call_OI'] = df[call_oi_col].apply(clean_val)
    df['Put_OI'] = df[put_oi_col].apply(clean_val)
    df['Call_IV'] = df[call_iv_col].apply(clean_val) / 100.0
    df['Put_IV'] = df[put_iv_col].apply(clean_val) / 100.0
            
    df = df.fillna(0)
    df['IV'] = df[['Call_IV', 'Put_IV']].mean(axis=1)
    
    expiry_info = "Unknown"
    data_date = "Unknown"
    basename = os.path.basename(sb_path)
    
    if "exp-" in basename:
        expiry_info = basename.split("exp-")[1].split("-")[0]
        
    # ファイル名末尾の日付 (例: 07-08-2026) を抽出
    date_match = re.search(r'(\d{2}-\d{2}-\d{4})\.csv$', basename)
    if date_match:
        data_date = date_match.group(1)
        
    return df.sort_values("Strike", ascending=False), expiry_info, data_date

def fetch_futures_spot(ticker):
    tkr = yf.Ticker(ticker)
    hist = tkr.history(period="1d")
    if hist.empty: raise ValueError(f"取得失敗: {ticker}")
    return hist['Close'].iloc[-1]

def calculate_gex(df, spot, multiplier):
    df = df[(df['Call_OI'] > 0) | (df['Put_OI'] > 0)].copy()
    T, r = 22 / 365.0, 0.045
    iv = np.where(df["IV"] <= 0.01, 0.01, df["IV"])
    
    d1 = (np.log(spot / df["Strike"]) + (r + iv**2 / 2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(T))
    
    df["Call_GEX"] = (df["Call_OI"] * gamma * spot * multiplier * 0.01) / 1e6
    df["Put_GEX"] = (-df["Put_OI"] * gamma * spot * multiplier * 0.01) / 1e6
    df["Net_GEX"] = df["Call_GEX"] + df["Put_GEX"]
    return df

def extract_flip_point(df, spot):
    df_sorted = df.sort_values("Strike").dropna(subset=["Net_GEX"])
    net_gex, strikes = df_sorted["Net_GEX"].values, df_sorted["Strike"].values
    sign_flips = np.where(np.diff(np.sign(net_gex)))[0]
    if len(sign_flips) == 0: return np.nan
    closest_flip_idx = min(sign_flips, key=lambda i: abs(strikes[i] - spot))
    x0, x1 = net_gex[closest_flip_idx], net_gex[closest_flip_idx + 1]
    y0, y1 = strikes[closest_flip_idx], strikes[closest_flip_idx + 1]
    if x1 - x0 == 0: return y0
    return y0 - (x0 * (y1 - y0) / (x1 - x0))

# --- 【新規追加】Gemini APIによる自動分析生成関数 ---
def generate_ai_insight(config, spot, flip_point, df):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "<p style='color: #ff4444;'>[System] GEMINI_API_KEYが設定されていないため、AIインサイトの生成はスキップされました。</p>"
    
    genai.configure(api_key=api_key)
    # 応答速度とコストのバランスから gemini-1.5-flash または pro を使用
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # 上下3つの最大の壁を抽出
    call_walls = df[df['Call_GEX'] > 0].nlargest(3, 'Call_GEX')[['Strike', 'Call_GEX']].to_dict('records')
    put_walls = df[df['Put_GEX'] < 0].nsmallest(3, 'Put_GEX')[['Strike', 'Put_GEX']].to_dict('records')
    
    regime = "POSITIVE (押し目買い・レンジ優位)" if spot > flip_point else "NEGATIVE (ブレイクアウト・順張り優位)"
    
    prompt = f"""
    あなたは金融工学とオプション取引に精通した「リード・クオンツアナリスト」です。
    以下の最新のGEX（ガンマ・エクスポージャー）データに基づき、プロのCFDトレーダー向けの簡潔な相場分析と実践的なトレード戦略を出力してください。
    
    【対象銘柄】: {config['name']}
    【現在価格 (Spot)】: {spot:.3f}
    【Zero-Gamma境界線】: {flip_point:.3f}
    【現在のレジーム】: {regime}
    
    【主要なレジスタンス (Call Wall)】
    1. Strike {call_walls[0]['Strike']} (GEX: {call_walls[0]['Call_GEX']:.2f}M)
    2. Strike {call_walls[1]['Strike']} (GEX: {call_walls[1]['Call_GEX']:.2f}M)
    
    【主要なサポート (Put Wall)】
    1. Strike {put_walls[0]['Strike']} (GEX: {put_walls[0]['Put_GEX']:.2f}M)
    2. Strike {put_walls[1]['Strike']} (GEX: {put_walls[1]['Put_GEX']:.2f}M)

    【出力要件】
    - 出力は必ずそのままWebに埋め込める **HTMLの断片のみ** とすること。（Markdownのコードブロック ```html などは絶対に含めないでください）
    - 以下のHTMLタグを駆使して、視覚的に見やすく構造化すること: <h3>, <ul>, <li>, <strong>, <br>
    - デザインテーマ（ダークモード、ハッカーライク）に合うよう、重要な数値や方向性にはインラインCSSで色付けをすること。（例: <strong style="color:#00FF00;">ロング</strong>, <strong style="color:#FF4444;">ショート</strong>, <span style="color:#00FFFF;">1200の壁</span> など）
    - 初心者向けの解説は不要。現在のレジームに基づく「どこでエントリーし、どこで利確・損切りすべきか」の具体的なアクションプランにフォーカスすること。
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text
        # Markdownのコードブロックタグが混入した場合の除去処理
        text = text.replace('```html', '').replace('```', '').strip()
        return text
    except Exception as e:
        return f"<p style='color: #ff4444;'>[Error] AIインサイト生成に失敗しました: {e}</p>"

def export_dashboard(df, spot, expiry, data_date, output_path, config):
    flip_point = extract_flip_point(df, spot)
    
    # --- AIインサイトの生成 ---
    print(f"[{config['ticker']}] Generating AI Insight via Gemini API...")
    ai_insight_html = generate_ai_insight(config, spot, flip_point, df)

    df_zoom = df[(df['Strike'] >= spot * 0.7) & (df['Strike'] <= spot * 1.3)].copy()
    
    iv_series = df_zoom["IV"].replace(0, np.nan)
    iv_median = iv_series.median()
    df_zoom["IV_plot"] = np.where(iv_series > iv_median * 2.5, np.nan, iv_series)
    
    is_positive = spot > flip_point
    regime_text = "🟢 POSITIVE GAMMA REGIME (押し目買い優位)" if is_positive else "🔴 NEGATIVE GAMMA REGIME (ブレイクアウト・順張り優位)"
    regime_color = "#00FF00" if is_positive else "#FF4444"
    
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.7, 0.3],
        subplot_titles=(f"Dealer Net GEX Profile<br><b style='color:{regime_color}; font-size:16px;'>{regime_text}</b>", "Implied Volatility Smile"),
        shared_xaxes=True, vertical_spacing=0.07
    )
    
    fig.add_trace(go.Bar(x=df_zoom["Strike"], y=df_zoom["Call_GEX"], name="Call GEX (レジスタンス)", marker_color="rgba(0, 255, 255, 0.7)"), row=1, col=1)
    fig.add_trace(go.Bar(x=df_zoom["Strike"], y=df_zoom["Put_GEX"], name="Put GEX (サポート)", marker_color="rgba(255, 0, 255, 0.7)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["Net_GEX"], name="Net GEX", mode="lines+markers", line=dict(color="white", width=2)), row=1, col=1)
    
    if not np.isnan(flip_point):
        fig.add_vline(x=flip_point, line=dict(color="red", width=2, dash="dashdot"), 
                      annotation_text=f"Zero-Gamma<br>{flip_point:.3f}", 
                      annotation_position="top left", 
                      annotation=dict(font=dict(color="white", size=13), bgcolor="rgba(255,0,0,0.6)", bordercolor="red", borderwidth=1), row=1, col=1)

    fig.add_vline(x=spot, line=dict(color="yellow", width=2, dash="solid"), 
                  annotation_text=f"Current Spot<br>{spot:.3f}", 
                  annotation_position="bottom right", 
                  annotation=dict(font=dict(color="black", size=13), bgcolor="rgba(255,255,0,0.8)", bordercolor="yellow", borderwidth=1), row=1, col=1)
        
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["IV_plot"]*100, name="IV", mode="lines+markers", line_color="orange", connectgaps=True), row=2, col=1)
    
    fig.update_layout(
        title=f"Quant Options Radar: {config['name']} | Expiry: {expiry}<br><span style='font-size: 13px; color: #aaaaaa;'>As of: {data_date}</span>",
        template="plotly_dark", 
        height=850,
        margin=dict(t=120),
        barmode='relative', hovermode='x unified',
        dragmode=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="GEX ($M)", row=1, col=1)
    fig.update_yaxes(title_text="IV (%)", row=2, col=1)
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)
    
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True, config={'displayModeBar': False})

    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # --- AIパネル用のCSSを追加 ---
    custom_head = """
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        body { margin: 0; background-color: #111; font-family: sans-serif; overflow-x: hidden; }
        .nav-bar {
            padding: 10px; background-color: #111; text-align: center; 
            border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 9999;
        }
        .nav-bar a { margin: 0 10px; text-decoration: none; font-weight: bold; font-size: 15px; display: inline-block; padding: 5px; }
        .mobile-notice {
            display: none; background-color: #2c3e50; color: #f1c40f; 
            text-align: center; padding: 8px; font-size: 13px; font-weight: bold;
        }
        .chart-scroll-container {
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        .chart-wrapper {
            width: 100%;
            min-width: 1100px;
            margin: 0 auto;
        }
        /* --- AIパネルのデザイン --- */
        .ai-panel-container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto 40px auto;
            padding: 0 15px;
            box-sizing: border-box;
        }
        .ai-panel {
            background-color: #1a1a1a;
            border-left: 4px solid #f9ab00;
            border-top: 1px solid #333;
            border-right: 1px solid #333;
            border-bottom: 1px solid #333;
            border-radius: 4px;
            padding: 20px;
            color: #dcdcdc;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
        }
        .ai-panel-header {
            color: #f9ab00;
            font-size: 14px;
            font-weight: bold;
            letter-spacing: 1px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }
        .ai-panel-header span {
            margin-right: 8px;
        }
        .ai-panel h3 { color: #fff; margin-top: 20px; font-size: 16px; border-bottom: 1px dotted #444; padding-bottom: 5px; }
        .ai-panel ul { padding-left: 20px; }
        .ai-panel li { margin-bottom: 8px; font-size: 14px; }
        
        @media screen and (max-width: 800px) {
            .mobile-notice { display: block; }
            .nav-bar a { font-size: 12px; margin: 0 3px; }
            .ai-panel { padding: 15px; }
        }
    </style>
    """
    
    # --- グラフの下にAIインサイトパネルを注入 ---
    ai_panel_html = f"""
        </div>
    </div> <!-- chart-scroll-container end -->
    
    <div class="ai-panel-container">
        <div class="ai-panel">
            <div class="ai-panel-header"><span>●</span> DAILY QUANT INSIGHT (Powered by Gemini AI)</div>
            {ai_insight_html}
        </div>
    </div>
    """
    
    nav_and_container = """
    <body>
    <div class="nav-bar">
        <a href="index.html" style="color: #00FFFF;">🪙 Silver (SI)</a>
        <a href="ng.html" style="color: #FF00FF;">🔥 Natural Gas (NG)</a>
        <a href="hg.html" style="color: #FF8C00;">🥉 Copper (HG)</a>
        <a href="zs.html" style="color: #32CD32;">🌱 Soybeans (ZS)</a>
        <a href="zc.html" style="color: #FFD700;">🌽 Corn (ZC)</a>
        <a href="zw.html" style="color: #DAA520;">🌾 Wheat (ZW)</a>
        <a href="gex_trading_guide.html" style="color: #FFFF00;">📖 Trading Manual</a>
    </div>
    <div class="mobile-notice">📱 グラフを左右にスワイプして詳細を確認できます</div>
    <div class="chart-scroll-container">
        <div class="chart-wrapper">
    """
    
    html_content = html_content.replace('<head>', f'<head>\n{custom_head}')
    html_content = html_content.replace('<body>', nav_and_container)
    # </body>タグの直前にAIパネルとスクリプトを挿入
    html_content = html_content.replace('</body>', f'{ai_panel_html}\n<script>\nwindow.addEventListener("load", function() {{\nvar container = document.querySelector(".chart-scroll-container");\nif (container) {{ container.scrollLeft = (container.scrollWidth - container.clientWidth) / 2; }}\n}});\n</script>\n</body>')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

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
