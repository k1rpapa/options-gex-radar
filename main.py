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
    "CT": {"name": "🧵 コットン (CT)", "ticker": "CT=F", "multiplier": 500, "filename": "ct.html"},
    "HE": {"name": "🐷 豚肉 (HE)", "ticker": "HE=F", "multiplier": 400, "filename": "he.html"},
    "CC": {"name": "🍫 ココア (CC)", "ticker": "CC=F", "multiplier": 10, "filename": "cc.html"},
    "KC": {"name": "☕ コーヒー (KC)", "ticker": "KC=F", "multiplier": 375, "filename": "kc.html"},
    "DX": {"name": "💵 ドルインデックス (DXY)", "ticker": "DX-Y.NYB", "multiplier": 1000, "filename": "dxy.html"},
    "J6": {"name": "💴 日本円 (JPY)", "ticker": "6J=F", "multiplier": 12500000, "filename": "jpy.html"}
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

def is_valid_greeks_file(fpath):
    """満期消滅等でGammaが全ストライク0になっている無効ファイルを判定"""
    try:
        df = pd.read_csv(fpath)
        gamma_cols = [c for c in df.columns if 'Gamma' in c]
        if not gamma_cols:
            return True
        for col in gamma_cols:
            vals = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('s', '').str.replace('%', ''), errors='coerce').fillna(0)
            if vals.max() > 0:
                return True
        return False
    except:
        return False

def get_all_csv_pairs(asset_key):
    """銘柄に該当する全日付のCSVペア (expiry, as_of) -> (sb_path, gk_path) を取得"""
    prefix = asset_key.lower()
    sb_pattern = re.compile(rf'^{prefix}[a-z0-9]*-options-.*exp-(\d{{2}}_\d{{2}}_\d{{2}}).*-(\d{{2}}-\d{{2}}-\d{{4}})(?: \(\d+\))?\.csv$', re.IGNORECASE)
    gk_pattern = re.compile(rf'^{prefix}[a-z0-9]*-volatility-greeks.*exp-(\d{{2}}_\d{{2}}_\d{{2}}).*-(\d{{2}}-\d{{2}}-\d{{4}})(?: \(\d+\))?\.csv$', re.IGNORECASE)
    
    sb_dict = {}
    gk_dict = {}
    
    for p in glob.glob(f"{prefix}*.csv"):
        fname = Path(p).name
        m_sb = sb_pattern.match(fname)
        if m_sb:
            key = (m_sb.group(1), m_sb.group(2))
            is_show_all = 'show-all' in fname
            if key not in sb_dict or (is_show_all and 'show-all' not in sb_dict[key]):
                sb_dict[key] = p
                
        m_gk = gk_pattern.match(fname)
        if m_gk:
            key = (m_gk.group(1), m_gk.group(2))
            is_show_all = 'show-all' in fname
            if key not in gk_dict or (is_show_all and 'show-all' not in gk_dict[key]):
                gk_dict[key] = p
                
    common_keys = sorted(
        [k for k in (set(sb_dict.keys()) & set(gk_dict.keys())) if is_valid_greeks_file(gk_dict[k])], 
        key=lambda k: (datetime.strptime(k[1], '%m-%d-%Y'), datetime.strptime(k[0], '%m_%d_%y'))
    )
    return common_keys, sb_dict, gk_dict

def load_barchart_csv(asset_key):
    common_keys, sb_dict, gk_dict = get_all_csv_pairs(asset_key)
    if not common_keys:
        # 万一完全一致ペアがない場合のフォールバック
        prefix = asset_key.lower()
        sb_files = glob.glob(f"{prefix}*side-by-side*.csv")
        gk_files = glob.glob(f"{prefix}*volatility-greeks*.csv")
        if not sb_files or not gk_files:
            return None, None, None, None
            
        def extract_date(fpath):
            dm = re.search(r'-(\d{2}-\d{2}-\d{4})(?: \(\d+\))?\.csv', fpath)
            if dm:
                try:
                    return datetime.strptime(dm.group(1), '%m-%d-%Y')
                except:
                    pass
            return datetime.min
            
        sb_path = max(sb_files, key=extract_date)
        gk_path = max(gk_files, key=extract_date)
        
        match = re.search(r'exp-(\d{2}_\d{2}_\d{2})', sb_path)
        expiry = match.group(1) if match else "Unknown"
        date_match = re.search(r'-(\d{2}-\d{2}-\d{4})\.csv', sb_path)
        as_of_date = date_match.group(1) if date_match else None
        
        df_sb = pd.read_csv(sb_path)
        df_gk = pd.read_csv(gk_path)
        return df_sb, df_gk, expiry, as_of_date
        
    best_key = common_keys[-1]
    sb_path = sb_dict[best_key]
    gk_path = gk_dict[best_key]
    
    df_sb = pd.read_csv(sb_path)
    df_gk = pd.read_csv(gk_path)
    
    expiry = best_key[0]
    as_of_date = best_key[1]
    
    return df_sb, df_gk, expiry, as_of_date

def extract_implied_spot_from_chain(df_sb, df_gk):
    """
    オプションチェーンデータからインプライド原資産価格（Forward/Spot）を高精度に逆算
    1. GreeksファイルのCall Deltaが0.50に最も近いストライク（Delta ATM Crossing）
    2. Side-by-Sideファイルのプットコールパリティ（S = K + C - P）
    """
    try:
        df_sb_c = df_sb.copy()
        df_gk_c = df_gk.copy()
        df_sb_c.columns = [str(c).strip() for c in df_sb_c.columns]
        df_gk_c.columns = [str(c).strip() for c in df_gk_c.columns]
        
        df_sb_c['Strike'] = df_sb_c['Strike'].apply(parse_strike)
        df_gk_c['Strike'] = df_gk_c['Strike'].apply(parse_strike)
        
        # --- 1. Delta 0.50 反転点 (ATM) の推定 ---
        delta_idx = [i for i, col in enumerate(df_gk_c.columns) if 'Delta' in col]
        delta_spot = None
        if len(delta_idx) >= 1:
            c_delta = pd.to_numeric(df_gk_c.iloc[:, delta_idx[0]].astype(str).str.replace(',', '').str.replace('s', ''), errors='coerce').fillna(0)
            valid_delta = df_gk_c[(c_delta > 0.05) & (c_delta < 0.95)]
            if not valid_delta.empty:
                idx_near = (c_delta.loc[valid_delta.index] - 0.50).abs().idxmin()
                delta_spot = float(df_gk_c.loc[idx_near, 'Strike'])
            else:
                # 満期直前等でDeltaが1と0に二極化している場合、境界ストライクを特定
                ones = df_gk_c[c_delta >= 0.90]
                zeros = df_gk_c[c_delta <= 0.10]
                if not ones.empty and not zeros.empty:
                    last_one = ones['Strike'].max()
                    first_zero = zeros[zeros['Strike'] > last_one]['Strike'].min() if (zeros['Strike'] > last_one).any() else last_one
                    delta_spot = (last_one + first_zero) / 2.0
                elif not ones.empty:
                    delta_spot = float(ones['Strike'].max())

        # --- 2. Put-Call Parity: S = K + C - P ---
        # side-by-side形式: Call_Latest(col 1), Strike(col 5), Put_Latest(col 7)
        if len(df_sb_c.columns) >= 8:
            call_latest = pd.to_numeric(df_sb_c.iloc[:, 1].astype(str).str.replace(',', '').str.replace('s', ''), errors='coerce').fillna(0)
            put_latest = pd.to_numeric(df_sb_c.iloc[:, 7].astype(str).str.replace(',', '').str.replace('s', ''), errors='coerce').fillna(0)
            
            valid_mask = (call_latest > 0) & (put_latest > 0) & (df_sb_c['Strike'] > 0)
            if valid_mask.any():
                df_valid = df_sb_c[valid_mask].copy()
                c_v = call_latest[valid_mask]
                p_v = put_latest[valid_mask]
                df_valid['Implied_Spot'] = df_valid['Strike'] + c_v - p_v
                
                # Delta推定値近傍（ATM ±20%）に絞ってノイズ排除
                if delta_spot and delta_spot > 0:
                    near_atm = df_valid[(df_valid['Strike'] >= delta_spot * 0.8) & (df_valid['Strike'] <= delta_spot * 1.2)]
                    if not near_atm.empty:
                        return float(near_atm['Implied_Spot'].median())
                return float(df_valid['Implied_Spot'].median())
                
        if delta_spot and delta_spot > 0:
            return float(delta_spot)
            
    except Exception as e:
        print(f"[-] Warning: Failed to extract implied spot from chain: {e}")
        
    return None

def resolve_spot_price(df_sb, df_gk, ticker_hist, file_as_of):
    """
    オプションチェーンのインプライド原資産価格とyfinance価格を照合し、最も信頼できるスポット価格を決定
    """
    implied_spot = extract_implied_spot_from_chain(df_sb, df_gk)
    
    yf_spot = None
    if ticker_hist is not None and not ticker_hist.empty and 'Close' in ticker_hist.columns:
        valid_close = ticker_hist['Close'].dropna()
        if not valid_close.empty:
            if file_as_of:
                try:
                    d = datetime.strptime(file_as_of, '%m-%d-%Y')
                    d_fmt = d.strftime('%Y-%m-%d')
                    exact_match = valid_close[valid_close.index.strftime('%Y-%m-%d') == d_fmt]
                    if not exact_match.empty:
                        yf_spot = float(exact_match.iloc[0])
                    else:
                        past_match = valid_close[valid_close.index.strftime('%Y-%m-%d') <= d_fmt]
                        if not past_match.empty:
                            yf_spot = float(past_match.iloc[-1])
                        else:
                            yf_spot = float(valid_close.iloc[-1])
                except:
                    yf_spot = float(valid_close.iloc[-1])
            else:
                yf_spot = float(valid_close.iloc[-1])

    # 優先順位決定:
    # 1. implied_spotが算出できている場合:
    #    - yf_spotが存在し、かつ乖離が5%以内なら限月直結のimplied_spotを採用（サヤズレ解消）
    #    - yf_spotがNaNや異常値（大幅乖離）の場合も、チェーン内の真の原資産価格implied_spotを採用
    if implied_spot and implied_spot > 0:
        if yf_spot and yf_spot > 0:
            diff_pct = abs(implied_spot - yf_spot) / yf_spot
            if diff_pct > 0.05:
                print(f"[*] Note: Discrepancy detected (Implied: {implied_spot:.3f}, YF: {yf_spot:.3f}, Diff: {diff_pct*100:.1f}%). Prioritizing Option-Implied Spot.")
        return float(implied_spot)
        
    # 2. implied_spotが算出できない場合のフォールバック: 有効なyf_spot
    if yf_spot and yf_spot > 0 and not np.isnan(yf_spot):
        return float(yf_spot)
        
    # 3. 万一すべて取得できない場合の最終防衛ライン: 0ではなく1.0（エラー防止）
    return 1.0

def calculate_gex_metrics(df_sb, df_gk, mult, spot_price_hint=None):
    """単一スナップショットからGEX、ZG、Call/Put Wallを算出"""
    df_sb = df_sb.copy()
    df_gk = df_gk.copy()
    df_sb.columns = [str(c).strip() for c in df_sb.columns]
    df_gk.columns = [str(c).strip() for c in df_gk.columns]
    
    df_sb['Strike'] = df_sb['Strike'].apply(parse_strike)
    df_gk['Strike'] = df_gk['Strike'].apply(parse_strike)
    
    oi_idx = [i for i, col in enumerate(df_sb.columns) if 'Open Int' in col or 'OI' in col]
    if len(oi_idx) >= 2:
        df_sb['Call_OpenInt'] = df_sb.iloc[:, oi_idx[0]].apply(clean_val)
        df_sb['Put_OpenInt'] = df_sb.iloc[:, oi_idx[1]].apply(clean_val)
        df_sb_agg = df_sb.groupby('Strike', as_index=False)[['Call_OpenInt', 'Put_OpenInt']].sum()
    elif 'Type' in df_sb.columns and len(oi_idx) == 1:
        calls = df_sb[df_sb['Type'].astype(str).str.lower() == 'call'].copy()
        puts = df_sb[df_sb['Type'].astype(str).str.lower() == 'put'].copy()
        calls['Call_OpenInt'] = calls.iloc[:, oi_idx[0]].apply(clean_val)
        puts['Put_OpenInt'] = puts.iloc[:, oi_idx[0]].apply(clean_val)
        c_agg = calls.groupby('Strike', as_index=False)['Call_OpenInt'].sum()
        p_agg = puts.groupby('Strike', as_index=False)['Put_OpenInt'].sum()
        df_sb_agg = pd.merge(c_agg, p_agg, on='Strike', how='outer').fillna(0.0)
    else:
        df_sb_agg = df_sb.groupby('Strike', as_index=False).agg({'Strike': 'first'})
        df_sb_agg['Call_OpenInt'] = 0.0
        df_sb_agg['Put_OpenInt'] = 0.0

    gamma_idx = [i for i, col in enumerate(df_gk.columns) if 'Gamma' in col]
    iv_idx = [i for i, col in enumerate(df_gk.columns) if 'IV' in col and 'Skew' not in col]

    if len(gamma_idx) >= 2 and len(iv_idx) >= 2:
        df_gk['Gamma_Call'] = df_gk.iloc[:, gamma_idx[0]].apply(clean_val)
        df_gk['Gamma_Put'] = df_gk.iloc[:, gamma_idx[1]].apply(clean_val)
        df_gk['IV_Call'] = df_gk.iloc[:, iv_idx[0]].apply(clean_val)
        df_gk['IV_Put'] = df_gk.iloc[:, iv_idx[1]].apply(clean_val)
        df_gk_agg = df_gk.groupby('Strike', as_index=False)[['Gamma_Call', 'Gamma_Put', 'IV_Call', 'IV_Put']].max()
    elif 'Type' in df_gk.columns and len(gamma_idx) >= 1:
        calls = df_gk[df_gk['Type'].astype(str).str.lower() == 'call'].copy()
        puts = df_gk[df_gk['Type'].astype(str).str.lower() == 'put'].copy()
        calls['Gamma_Call'] = calls.iloc[:, gamma_idx[0]].apply(clean_val)
        calls['IV_Call'] = calls.iloc[:, iv_idx[0]].apply(clean_val) if iv_idx else 0.0
        puts['Gamma_Put'] = puts.iloc[:, gamma_idx[0]].apply(clean_val)
        puts['IV_Put'] = puts.iloc[:, iv_idx[0]].apply(clean_val) if iv_idx else 0.0
        c_agg = calls.groupby('Strike', as_index=False)[['Gamma_Call', 'IV_Call']].max()
        p_agg = puts.groupby('Strike', as_index=False)[['Gamma_Put', 'IV_Put']].max()
        df_gk_agg = pd.merge(c_agg, p_agg, on='Strike', how='outer').fillna(0.0)
    else:
        df_gk_agg = df_gk.groupby('Strike', as_index=False).agg({'Strike': 'first'})
        df_gk_agg['Gamma_Call'] = 0.0
        df_gk_agg['Gamma_Put'] = 0.0
        df_gk_agg['IV_Call'] = 0.0
        df_gk_agg['IV_Put'] = 0.0

    df_merged = df_gk_agg.merge(df_sb_agg, on='Strike', how='outer').fillna(0)
    
    if spot_price_hint and spot_price_hint > 0 and not np.isnan(spot_price_hint):
        spot_price = float(spot_price_hint)
    else:
        spot_price = extract_implied_spot_from_chain(df_sb, df_gk) or 1.0
    if spot_price == 0.0:
        spot_price = 1.0

    spot_scale = (spot_price ** 2) * 0.01 / 1e6
    df_merged['Call_GEX'] = df_merged['Gamma_Call'] * df_merged['Call_OpenInt'] * mult * spot_scale
    df_merged['Put_GEX'] = df_merged['Gamma_Put'] * df_merged['Put_OpenInt'] * mult * spot_scale * -1
    df_merged['Total_GEX'] = df_merged['Call_GEX'] + df_merged['Put_GEX']
    df_merged['Total_OI'] = df_merged['Call_OpenInt'] + df_merged['Put_OpenInt']

    # 1. 全体チェーンでの主要ウォール算出 (Major / Minor Call & Put Walls)
    call_walls_df = df_merged[df_merged['Call_GEX'] > 0].nlargest(2, 'Call_GEX')[['Strike', 'Call_GEX']].copy()
    put_walls_df = df_merged[df_merged['Put_GEX'] < 0].nsmallest(2, 'Put_GEX')[['Strike', 'Put_GEX']].copy()
    
    call_walls = call_walls_df.to_dict('records')
    put_walls = put_walls_df.to_dict('records')
    while len(call_walls) < 2: call_walls.append({"Strike": 0.0, "Call_GEX": 0.0})
    while len(put_walls) < 2: put_walls.append({"Strike": 0.0, "Put_GEX": 0.0})

    # 2. 全体チェーンでの Zero Gamma (ZG) 算出 (ノイズ排除: Total_OI > max(Total_OI) * 0.05)
    max_oi = df_merged['Total_OI'].max()
    max_gex = df_merged['Total_GEX'].abs().max()
    if max_gex > 0:
        valid_mask = (df_merged['Total_OI'] > max_oi * 0.05) if max_oi > 0 else (df_merged['Total_OI'] > 0)
        df_valid = df_merged[valid_mask].sort_values('Strike').reset_index(drop=True)
        
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
                zero_gamma_strike = closest_flip_strike
            else:
                zero_gamma_idx = df_valid['Total_GEX'].abs().idxmin()
                zero_gamma_strike = df_valid.loc[zero_gamma_idx, 'Strike']
        else:
            zero_gamma_strike = spot_price
    else:
        zero_gamma_strike = spot_price

    # 3. グラフ描画用 (GEXラダー用) に Spot近傍のストライクを抽出
    min_strike = spot_price * 0.85
    max_strike = spot_price * 1.15
    df_near = df_merged[(df_merged['Strike'] >= min_strike) & (df_merged['Strike'] <= max_strike)].copy()
    active_mask = (df_near['Total_OI'] > 0) | (df_near['Total_GEX'].abs() > 0.001)
    df_active = df_near[active_mask].copy()

    if len(df_active) < 12:
        df_active = df_merged[(df_merged['Total_OI'] > 0) | (df_merged['Total_GEX'].abs() > 0.001)].copy()

    if df_active.empty:
        df_active = df_merged.copy()

    if len(df_active) > 32:
        df_active['Dist_From_Spot'] = (df_active['Strike'] - spot_price).abs()
        df_active = df_active.nsmallest(32, 'Dist_From_Spot').copy()

    df_sorted = df_active.sort_values('Strike').reset_index(drop=True)

    return {
        "df_sorted": df_sorted,
        "spot_price": spot_price,
        "zero_gamma": zero_gamma_strike,
        "call_walls": call_walls,
        "put_walls": put_walls
    }

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
    
    prompt = (
        "あなたは金融工学とオプション取引に精通した「リード・クオンツアナリスト」です。\n"
        "以下の全銘柄の最新GEXデータおよび【Zero-Gammaの直近急変・推移アラート】に基づき、オプション初心者にも分かるように現在の重力場の分析と、実践的なトレード戦略をJSONフォーマットで出力してください。\n\n"
        "【全銘柄のデータ】\n"
        f"{json.dumps(asset_summaries, ensure_ascii=False, indent=2)}\n\n"
        "【各銘柄のインサイト出力ルール (厳守事項)】\n"
        "- Webページに直接埋め込むため、各銘柄の値(Value)には純粋なHTMLの断片のみを文字列として出力すること。\n"
        "- 以下のHTMLタグを駆使して構造化すること: <h3>, <ul>, <li>, <p>, <strong>\n"
        "- ダークテーマのダッシュボードに映えるよう、重要な数値や方向性にはインラインCSSで色付けをすること。（例: <span style='color: #44c265;'>）\n"
        "- 「Call Wall（レジスタンス）」「Put Wall（サポート）」および「Zero Gammaの急変や移動」を具体的に引用しながら、現在のレジームに基づく「どこでエントリーし、どこで利確・損切りすべきか」の具体的なアクションプランにフォーカスすること。\n\n"
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

def build_timeline_dataframe(asset_key, config, ticker_hist):
    """過去全日のスナップショットから時系列DataFrameを構築"""
    common_keys, sb_dict, gk_dict = get_all_csv_pairs(asset_key)
    if not common_keys:
        return pd.DataFrame()
        
    records = []
    mult = config['multiplier']
    
    for exp, date_str in common_keys:
        d = datetime.strptime(date_str, '%m-%d-%Y')
        df_sb = pd.read_csv(sb_dict[(exp, date_str)])
        df_gk = pd.read_csv(gk_dict[(exp, date_str)])
        
        spot = resolve_spot_price(df_sb, df_gk, ticker_hist, date_str)
        metrics = calculate_gex_metrics(df_sb, df_gk, mult, spot)
        
        cw1 = metrics['call_walls'][0]['Strike']
        cw2 = metrics['call_walls'][1]['Strike']
        pw1 = metrics['put_walls'][0]['Strike']
        pw2 = metrics['put_walls'][1]['Strike']
        
        records.append({
            "date": d.strftime('%m/%d'),
            "full_date": date_str,
            "datetime": d,
            "spot": round(metrics['spot_price'], 4),
            "zero_gamma": round(metrics['zero_gamma'], 4),
            "call_wall_1": cw1,
            "call_wall_2": cw2,
            "put_wall_1": pw1,
            "put_wall_2": pw2
        })
        
    df_timeline = pd.DataFrame(records).sort_values('datetime').reset_index(drop=True)
    return df_timeline

def evaluate_zg_alert(df_timeline, current_spot, current_zg):
    """ZG急変およびレジーム変化を判定してアラート情報を生成"""
    if df_timeline.empty or len(df_timeline) < 2:
        return {
            "level": "info",
            "title": "ZG STABLE (通常推移)",
            "badge_class": "badge-stable",
            "delta_zg": 0.0,
            "pct_zg": 0.0,
            "message": "履歴データが蓄積され次第、急変検知を開始します。"
        }
        
    prev_row = df_timeline.iloc[-2]
    curr_row = df_timeline.iloc[-1]
    
    prev_zg = prev_row['zero_gamma']
    prev_spot = prev_row['spot']
    
    delta_zg = current_zg - prev_zg
    pct_zg = (delta_zg / prev_zg * 100.0) if prev_zg > 0 else 0.0
    
    prev_is_pos = prev_spot >= prev_zg
    curr_is_pos = current_spot >= current_zg
    regime_flipped = (prev_is_pos != curr_is_pos)
    
    fmt = lambda x: f"{x:.4f}" if x < 0.1 else (f"{x:.2f}" if x < 1000 else f"{x:.1f}")
    sign = "+" if delta_zg >= 0 else ""
    
    # 判定基準: 変動率 >= 5% または レジーム反転
    if abs(pct_zg) >= 5.0:
        direction = "急騰 🚀" if delta_zg > 0 else "急落 🔻"
        consequence = (
            "ZGが大幅に切り上がったため、SpotがNegative Gamma圏へ転落しやすく、ボラティリティ急拡大・下落スクイーズへの警戒が必要です。"
            if delta_zg > 0 else
            "ZGが急低下したため、Positive Gamma圏が拡大し、相場が安定・レンジ回帰（Gamma Pinning）しやすくなっています。"
        )
        return {
            "level": "critical",
            "title": f"ZG SPIKE ALERT: Zero-Gamma {direction}",
            "badge_class": "badge-critical",
            "delta_zg": delta_zg,
            "pct_zg": pct_zg,
            "message": f"ZGが直近比較で <strong>{sign}{fmt(delta_zg)} ({sign}{pct_zg:.1f}%)</strong> 大幅変動（{fmt(prev_zg)} ➔ {fmt(current_zg)}）。{consequence}"
        }
    elif regime_flipped:
        new_regime = "POSITIVE (押し目優位)" if curr_is_pos else "NEGATIVE (ボラ拡大警戒)"
        return {
            "level": "warning",
            "title": f"REGIME FLIP: ガンマ・レジーム転換 ➔ {new_regime}",
            "badge_class": "badge-warning",
            "delta_zg": delta_zg,
            "pct_zg": pct_zg,
            "message": f"Spot価格（{fmt(current_spot)}）がZG（{fmt(current_zg)}）と交差し、レジームが反転しました。トレード戦略の切り替えを推奨します。"
        }
    elif abs(pct_zg) >= 2.5:
        return {
            "level": "warning",
            "title": "ZG SHIFT: Zero-Gamma有意な変動検知",
            "badge_class": "badge-warning",
            "delta_zg": delta_zg,
            "pct_zg": pct_zg,
            "message": f"ZGが <strong>{sign}{fmt(delta_zg)} ({sign}{pct_zg:.1f}%)</strong> シフト（{fmt(prev_zg)} ➔ {fmt(current_zg)}）。ウォールの移動と防衛ラインを確認してください。"
        }
    else:
        return {
            "level": "info",
            "title": "ZG STABLE: 安定レジーム維持",
            "badge_class": "badge-stable",
            "delta_zg": delta_zg,
            "pct_zg": pct_zg,
            "message": f"Zero-Gammaは安定推移しています（前日比 {sign}{fmt(delta_zg)} / {sign}{pct_zg:.1f}%）。現在の防衛ラインが機能中です。"
        }

def generate_timeline_chart(df_timeline, asset_name):
    """案A: 階層別マルチライン＆ガンマレンジ・バンドの時系列チャートを生成"""
    if df_timeline.empty or len(df_timeline) < 1:
        return "<p style='color:#c4c7c5; padding:20px;'>時系列データを蓄積中です。</p>"
        
    fig = go.Figure()
    
    # 1. ガンマレンジ・バンド (Major Put Wall 〜 Major Call Wall)
    # 下限トレース（透明）
    fig.add_trace(go.Scatter(
        x=df_timeline['date'],
        y=df_timeline['put_wall_1'],
        mode='lines',
        line=dict(width=0, color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    # 上限トレース（塗りつぶし）
    fig.add_trace(go.Scatter(
        x=df_timeline['date'],
        y=df_timeline['call_wall_1'],
        mode='lines',
        line=dict(width=0, color='rgba(0,0,0,0)'),
        fill='tonexty',
        fillcolor='rgba(6, 187, 223, 0.07)',
        name='Gamma Range (想定取引レンジ)',
        hoverinfo='skip'
    ))
    
    # 2. Minor Walls (破線・太さ2.0・高視認性)
    fig.add_trace(go.Scatter(
        x=df_timeline['date'],
        y=df_timeline['call_wall_2'],
        mode='lines+markers',
        line=dict(color='rgba(6, 187, 223, 0.85)', width=2.0, dash='dash'),
        marker=dict(size=4, color='#06bbdf'),
        name='Minor Call Wall (第2抵抗線)',
        hovertemplate='Minor Call Wall: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_timeline['date'],
        y=df_timeline['put_wall_2'],
        mode='lines+markers',
        line=dict(color='rgba(197, 152, 255, 0.85)', width=2.0, dash='dash'),
        marker=dict(size=4, color='#c598ff'),
        name='Minor Put Wall (第2支持線)',
        hovertemplate='Minor Put Wall: %{y}<extra></extra>'
    ))
    
    # 3. Major Walls (太線実線 3.0)
    fig.add_trace(go.Scatter(
        x=df_timeline['date'],
        y=df_timeline['call_wall_1'],
        mode='lines+markers',
        line=dict(color='#06bbdf', width=3.0),
        marker=dict(size=6, color='#06bbdf'),
        name='Major Call Wall (主要抵抗線)',
        hovertemplate='Major Call Wall: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_timeline['date'],
        y=df_timeline['put_wall_1'],
        mode='lines+markers',
        line=dict(color='#c598ff', width=3.0),
        marker=dict(size=6, color='#c598ff'),
        name='Major Put Wall (主要支持線)',
        hovertemplate='Major Put Wall: %{y}<extra></extra>'
    ))
    
    # 4. Zero Gamma (ZG) (赤破線太線 2.8)
    fig.add_trace(go.Scatter(
        x=df_timeline['date'],
        y=df_timeline['zero_gamma'],
        mode='lines+markers',
        line=dict(color='#ff4d4f', width=2.8, dash='dash'),
        marker=dict(size=6, color='#ff4d4f'),
        name='Zero-Gamma (ZG)',
        hovertemplate='<b>Zero-Gamma: %{y}</b><extra></extra>'
    ))
    
    # 5. Spot Price (黄太実線 3.5)
    fig.add_trace(go.Scatter(
        x=df_timeline['date'],
        y=df_timeline['spot'],
        mode='lines+markers',
        line=dict(color='#f1c40f', width=3.5),
        marker=dict(size=7, color='#f1c40f'),
        name='Spot Price (原資産価格)',
        hovertemplate='<b>Spot Price: %{y}</b><extra></extra>'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101218",
        plot_bgcolor="#101218",
        hovermode="x unified",
        margin=dict(t=20, b=40, l=60, r=30),
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#c4c7c5"),
            bgcolor="rgba(16, 18, 24, 0.85)"
        )
    )
    fig.update_xaxes(title_text="Date (時系列推移)", gridcolor="#22252e", zerolinecolor="#333742")
    fig.update_yaxes(title_text="Price / Strike ($)", gridcolor="#22252e", zerolinecolor="#333742")
    
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

def process_asset_data(asset_key, config, ticker_hist):
    df_sb, df_gk, expiry, file_as_of = load_barchart_csv(asset_key)
    if df_sb is None:
        raise FileNotFoundError(f"CSV files not found for {asset_key}")
        
    mult = config['multiplier']
    
    spot_price = resolve_spot_price(df_sb, df_gk, ticker_hist, file_as_of)
    metrics = calculate_gex_metrics(df_sb, df_gk, mult, spot_price)
    df_sorted = metrics['df_sorted']
    spot_price = metrics['spot_price']
    zero_gamma_strike = metrics['zero_gamma']
    call_walls = metrics['call_walls']
    put_walls = metrics['put_walls']
    
    # 時系列データの構築
    df_timeline = build_timeline_dataframe(asset_key, config, ticker_hist)
    
    # ZG急変アラートの判定
    alert_info = evaluate_zg_alert(df_timeline, spot_price, zero_gamma_strike)
    
    spot_date = file_as_of if file_as_of else datetime.now().strftime('%m-%d-%Y')
    is_positive = spot_price > zero_gamma_strike
    regime_str = "POSITIVE GAMMA REGIME (押し目買い優位)" if is_positive else "NEGATIVE GAMMA REGIME (パニック売り警戒)"
    regime_color = "#44c265" if is_positive else "#fe8983"
    
    fmt = lambda x: f"{x:.5f}" if x < 0.1 else (f"{x:.3f}" if x < 1000 else f"{x:.2f}")

    data_summary = {
        "asset_name": config['name'],
        "spot": round(spot_price, 5) if spot_price < 0.1 else round(spot_price, 3),
        "zero_gamma": round(zero_gamma_strike, 5) if zero_gamma_strike < 0.1 else round(zero_gamma_strike, 3),
        "regime": regime_str,
        "call_walls": call_walls,
        "put_walls": put_walls,
        "zg_alert_level": alert_info["level"],
        "zg_alert_title": alert_info["title"],
        "zg_delta": round(alert_info["delta_zg"], 3),
        "zg_pct_change": round(alert_info["pct_zg"], 1)
    }
    
    # 本日のGEXラダー＆IVプロファイル (上段チャート)
    if len(df_sorted) > 1:
        diffs = df_sorted['Strike'].diff().dropna()
        diffs = diffs[diffs > 0]
        strike_gap = diffs.median() if not diffs.empty else spot_price * 0.01
    else:
        strike_gap = spot_price * 0.01
    bar_width = strike_gap * 0.85

    fig_ladder = make_subplots(
        rows=1, cols=2, 
        shared_yaxes=True, 
        horizontal_spacing=0.03, 
        column_widths=[0.72, 0.28],
        subplot_titles=("GEX Option Ladder ($M)", "IV Profile (%)")
    )
    
    # Put GEX Bar
    fig_ladder.add_trace(go.Bar(
        y=df_sorted['Strike'], 
        x=df_sorted['Put_GEX'], 
        orientation='h', 
        width=bar_width,
        name='Put GEX (サポート)', 
        marker=dict(color='rgba(197, 152, 255, 0.85)', line=dict(color='#c598ff', width=1)),
        hovertemplate='<b>Strike: %{y}</b><br>Put GEX: $%{x:.2f}M<br>Put OI: %{customdata[0]:,.0f}<extra></extra>',
        customdata=df_sorted[['Put_OpenInt']].values
    ), row=1, col=1)
    
    # Call GEX Bar
    fig_ladder.add_trace(go.Bar(
        y=df_sorted['Strike'], 
        x=df_sorted['Call_GEX'], 
        orientation='h', 
        width=bar_width,
        name='Call GEX (レジスタンス)', 
        marker=dict(color='rgba(6, 187, 223, 0.85)', line=dict(color='#06bbdf', width=1)),
        hovertemplate='<b>Strike: %{y}</b><br>Call GEX: $%{x:.2f}M<br>Call OI: %{customdata[0]:,.0f}<extra></extra>',
        customdata=df_sorted[['Call_OpenInt']].values
    ), row=1, col=1)
    
    # Net GEX Line + Markers
    fig_ladder.add_trace(go.Scatter(
        y=df_sorted['Strike'], 
        x=df_sorted['Total_GEX'], 
        mode='lines+markers', 
        name='Net GEX', 
        line=dict(color='#ffffff', width=2.5), 
        marker=dict(size=5, color='#ffffff'),
        hovertemplate='<b>Strike: %{y}</b><br>Net GEX: $%{x:.2f}M<extra></extra>'
    ), row=1, col=1)
    
    # Current Spot Line
    fig_ladder.add_hline(
        y=spot_price, line_width=2, line_dash="solid", line_color="#f1c40f", 
        row=1, col=1, 
        annotation_text=f"Current Spot: {fmt(spot_price)}", 
        annotation_position="top right", 
        annotation_bgcolor="#f1c40f", 
        annotation_font_color="#000000",
        annotation_font_size=11
    )
    
    # Zero Gamma Line
    fig_ladder.add_hline(
        y=zero_gamma_strike, line_width=1.5, line_dash="dashdot", line_color="#ff4d4f", 
        row=1, col=1, 
        annotation_text=f"Zero-Gamma: {fmt(zero_gamma_strike)}", 
        annotation_position="bottom left", 
        annotation_bgcolor="#ff4d4f", 
        annotation_font_color="#ffffff",
        annotation_font_size=11
    )
    
    # IV Profile Trace
    df_sorted['IV_Avg'] = (df_sorted['IV_Call'] + df_sorted['IV_Put']) / 2
    df_iv = df_sorted[df_sorted['IV_Avg'] > 0]
    fig_ladder.add_trace(go.Scatter(
        y=df_iv['Strike'], 
        x=df_iv['IV_Avg'], 
        mode='lines+markers', 
        name='IV', 
        line=dict(color='#ff9f43', width=2.5), 
        marker=dict(size=5, color='#ff9f43'),
        hovertemplate='<b>Strike: %{y}</b><br>IV: %{x:.1f}%<extra></extra>'
    ), row=1, col=2)
    
    fig_ladder.update_layout(
        template="plotly_dark", 
        paper_bgcolor="#101218", 
        plot_bgcolor="#101218",
        barmode='overlay', 
        bargap=0.15,
        hovermode="y unified",
        showlegend=False,
        margin=dict(t=30, b=40, l=65, r=40),
        height=620
    )
    fig_ladder.update_yaxes(title_text="Strike Price", row=1, col=1, gridcolor="#22252e", zerolinecolor="#333742")
    fig_ladder.update_xaxes(title_text="GEX ($M)", row=1, col=1, gridcolor="#22252e", zerolinecolor="#333742")
    fig_ladder.update_yaxes(gridcolor="#22252e", row=1, col=2)
    fig_ladder.update_xaxes(title_text="IV (%)", row=1, col=2, gridcolor="#22252e", zerolinecolor="#333742")

    ladder_html = fig_ladder.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
    
    # 時系列チャート (下段)
    timeline_html = generate_timeline_chart(df_timeline, config['name'])
    
    header_info = {
        "expiry": expiry,
        "spot_date": spot_date,
        "regime_str": regime_str,
        "regime_color": regime_color,
        "is_positive": is_positive,
        "alert_info": alert_info
    }
    return ladder_html, timeline_html, data_summary, header_info

def main():
    ladder_graphs = {}
    timeline_graphs = {}
    asset_summaries = {}
    headers_info = {}
    
    # 事前に全銘柄の株価ヒストリを取得
    ticker_hists = {}
    print("[*] Pre-fetching market historical prices...")
    for key, config in ASSET_CONFIG.items():
        try:
            ticker_hists[key] = yf.Ticker(config['ticker']).history(period="6mo")
        except Exception as e:
            print(f"[-] Warning: Failed to fetch history for {config['ticker']}: {e}")
            ticker_hists[key] = pd.DataFrame()
            
    for key, config in ASSET_CONFIG.items():
        print(f"[*] Processing {config['name']}...")
        try:
            ladder_html, timeline_html, summary, header_info = process_asset_data(key, config, ticker_hists.get(key, pd.DataFrame()))
            ladder_graphs[key] = ladder_html
            timeline_graphs[key] = timeline_html
            asset_summaries[key] = summary
            headers_info[key] = header_info
        except Exception as e:
            print(f"[-] Error: Failed to process {config['name']}: {e}")
            ladder_graphs[key] = f"<p style='color:red;'>データ処理エラー: {e}</p>"
            timeline_graphs[key] = ""

    print("\n[*] Dispatching batch request to Gemini AI...")
    ai_insights = {}
    if asset_summaries:
        ai_insights = generate_batched_insights(asset_summaries)

    for key, config in ASSET_CONFIG.items():
        print(f"[*] Building HTML for {config['name']}...")
        ladder_html = ladder_graphs.get(key, "")
        timeline_html = timeline_graphs.get(key, "")
        
        info = headers_info.get(key, {
            "expiry": "N/A", "spot_date": "N/A", 
            "regime_str": "UNKNOWN", "regime_color": "#ffffff", "is_positive": True,
            "alert_info": {"level": "info", "title": "STABLE", "badge_class": "badge-stable", "message": ""}
        })
        alert = info.get("alert_info", {"level": "info", "title": "STABLE", "badge_class": "badge-stable", "message": ""})
        
        insight_content = ai_insights.get(key, "<p style='color:#fe8983;'>インサイトデータの取得に失敗しました。</p>")
        if isinstance(insight_content, dict):
            insight_content = insight_content.get("error", str(insight_content))

        tabs_links = []
        for k, cfg in ASSET_CONFIG.items():
            active_cls = "active" if k == key else ""
            tabs_links.append(f'<a href="{cfg["filename"]}" class="{active_cls}">{cfg["name"]}</a>')
        tabs_html = "\n                ".join(tabs_links)

        regime_bg = "rgba(68, 194, 101, 0.15)" if info['is_positive'] else "rgba(254, 137, 131, 0.15)"
        
        # アラートバナーのスタイル設定
        alert_bg = "rgba(11, 87, 208, 0.12)"
        alert_border = "#0b57d0"
        alert_icon = "🟢"
        if alert['level'] == 'critical':
            alert_bg = "rgba(255, 77, 79, 0.18)"
            alert_border = "#ff4d4f"
            alert_icon = "🚨"
        elif alert['level'] == 'warning':
            alert_bg = "rgba(249, 171, 0, 0.18)"
            alert_border = "#f9ab00"
            alert_icon = "⚠️"

        html_content = (
            '<!DOCTYPE html>\n'
            '<html lang="ja" data-theme="dark">\n'
            '<head>\n'
            '    <meta charset="UTF-8">\n'
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'    <title>Quant GEX Radar - {config["name"]}</title>\n'
            '    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
            '    <style>\n'
            '        * { box-sizing: border-box; }\n'
            '        body { background-color: #101218; color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 0; }\n'
            '        .nav-tabs { background: #1a1d21; padding: 10px; display: flex; flex-wrap: wrap; gap: 8px; position: sticky; top: 0; z-index: 100; border-bottom: 1px solid #2d2f38; }\n'
            '        .nav-tabs a { color: #c4c7c5; text-decoration: none; padding: 8px 16px; border-radius: 4px; font-size: 14px; white-space: nowrap; transition: all 0.2s ease; }\n'
            '        .nav-tabs a:hover { background: #2d2f38; color: #ffffff; }\n'
            '        .nav-tabs a.active { background: #0b57d0; color: white; font-weight: bold; }\n'
            '        .container { max-width: 1800px; margin: 0 auto; padding: 20px 15px; width: 100%; display: flex; flex-direction: column; gap: 24px; }\n'
            '        \n'
            '        /* ZG Alert Banner */\n'
            f'        .zg-alert-card {{ background: {alert_bg}; border-left: 5px solid {alert_border}; border-top: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05); padding: 14px 20px; border-radius: 8px; display: flex; align-items: center; gap: 16px; }}\n'
            '        .zg-alert-icon { font-size: 26px; line-height: 1; flex-shrink: 0; }\n'
            '        .zg-alert-content { flex-grow: 1; font-size: 14px; line-height: 1.5; color: #e3e3e3; }\n'
            f'        .zg-alert-title {{ font-size: 15px; font-weight: bold; color: {alert_border}; margin-bottom: 4px; display: flex; align-items: center; gap: 8px; }}\n'
            '        \n'
            '        /* Upper 2-column Grid */\n'
            '        .dashboard-grid { display: flex; flex-direction: column; gap: 20px; width: 100%; }\n'
            '        .chart-panel { width: 100%; min-width: 0; background: #14171f; padding: 18px; border-radius: 10px; border: 1px solid #232734; }\n'
            '        .ai-panel { width: 100%; min-width: 0; background: #14171f; padding: 18px; border-radius: 10px; border: 1px solid #232734; }\n'
            '        .panel-header { margin-bottom: 12px; display: flex; flex-direction: column; gap: 8px; }\n'
            '        .chart-main-title { font-size: 18px; font-weight: bold; color: #ffffff; line-height: 1.3; }\n'
            '        .chart-sub-info { font-size: 12px; color: #c4c7c5; }\n'
            '        .regime-badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: bold; align-self: flex-start; }\n'
            '        .html-legend { display: flex; flex-wrap: wrap; gap: 15px; font-size: 12px; color: #c4c7c5; margin-top: 4px; }\n'
            '        .legend-item { display: flex; align-items: center; gap: 6px; }\n'
            '        .color-dot { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }\n'
            '        \n'
            '        @media (min-width: 992px) {\n'
            '            .dashboard-grid { flex-direction: row; align-items: stretch; justify-content: space-between; }\n'
            '            .chart-panel { width: 55%; flex: 0 0 55%; min-width: 0; }\n'
            '            .ai-panel { width: 43%; flex: 0 0 43%; min-width: 0; }\n'
            '        }\n'
            '        \n'
            '        .ai-header { color: #f9ab00; font-weight: bold; font-size: 16px; display: flex; align-items: center; gap: 8px; line-height: 1.3; margin-bottom: 0; }\n'
            '        .ai-content { background: #1a1d21; padding: 18px; border-radius: 8px; border-left: 4px solid #0b57d0; font-size: 14px; line-height: 1.6; color: #c4c7c5; margin-top: 8px; }\n'
            '        .ai-content h3 { color: #e0e0e0; font-size: 16px; border-bottom: 1px solid #333; padding-bottom: 8px; margin-top: 0; }\n'
            '        .ai-content ul { padding-left: 20px; }\n'
            '        .ai-content li { margin-bottom: 8px; }\n'
            '        \n'
            '        /* Lower Full-Width Timeline Panel (Plan A) */\n'
            '        .timeline-panel { width: 100%; background: #14171f; padding: 20px; border-radius: 10px; border: 1px solid #232734; }\n'
            '        .timeline-header { margin-bottom: 12px; display: flex; flex-direction: column; gap: 6px; }\n'
            '        .timeline-title { font-size: 17px; font-weight: bold; color: #ffffff; display: flex; align-items: center; gap: 8px; }\n'
            '        .timeline-sub { font-size: 12px; color: #8e918f; }\n'
            '        .timeline-legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: #c4c7c5; margin-top: 6px; }\n'
            '    </style>\n'
            '</head>\n'
            '<body>\n'
            '    <div class="nav-tabs">\n'
            f'        {tabs_html}\n'
            '        <a href="gex_trading_guide.html" style="margin-left:auto; color: #f9ab00;">■ 取引マニュアル</a>\n'
            '    </div>\n'
            '    <div class="container">\n'
            '        <!-- 1. ZG Alert Banner -->\n'
            '        <div class="zg-alert-card">\n'
            f'            <div class="zg-alert-icon">{alert_icon}</div>\n'
            '            <div class="zg-alert-content">\n'
            f'                <div class="zg-alert-title">{alert["title"]}</div>\n'
            f'                <div>{alert["message"]}</div>\n'
            '            </div>\n'
            '        </div>\n'
            '        \n'
            '        <!-- 2. Upper Row: Current Ladder & AI Insights -->\n'
            '        <div class="dashboard-grid">\n'
            '            <div class="chart-panel">\n'
            '                <div class="panel-header">\n'
            f'                    <div class="chart-main-title">Snapshot: {config["name"]}</div>\n'
            f'                    <div class="chart-sub-info">Expiry: {info["expiry"]} &nbsp;|&nbsp; As of: {info["spot_date"]}</div>\n'
            f'                    <div class="regime-badge" style="background:{regime_bg}; color:{info["regime_color"]}; border:1px solid {info["regime_color"]};">● {info["regime_str"]}</div>\n'
            '                    <div class="html-legend">\n'
            '                        <span class="legend-item"><span class="color-dot" style="background:#c598ff"></span>Put GEX (サポート)</span>\n'
            '                        <span class="legend-item"><span class="color-dot" style="background:#06bbdf"></span>Call GEX (レジスタンス)</span>\n'
            '                        <span class="legend-item"><span class="color-dot" style="background:#ffffff"></span>Net GEX</span>\n'
            '                        <span class="legend-item"><span class="color-dot" style="background:#ffa500"></span>IV</span>\n'
            '                    </div>\n'
            '                </div>\n'
            f'                {ladder_html}\n'
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
            '        \n'
            '        <!-- 3. Lower Row: Historical Timeline Chart (Plan A) -->\n'
            '        <div class="timeline-panel">\n'
            '            <div class="timeline-header">\n'
            f'                <div class="timeline-title">📈 GEX & Zero-Gamma Historical Timeline: {config["name"]}</div>\n'
            '                <div class="timeline-sub">過去のSpot価格推移、Zero-Gamma（ZG）、および主要コール/プットウォールの防衛線シフト（ガンマレンジ帯）</div>\n'
            '            </div>\n'
            f'            {timeline_html}\n'
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
