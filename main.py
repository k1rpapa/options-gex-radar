import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# パス定義
ROOT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = ROOT_DIR / "docs"

def load_barchart_csv():
    """リポジトリ内のBarchartのCSVを自動スキャンして結合・パースする"""
    # 1. ファイルの自動検知
    side_by_side_files = glob.glob("*side-by-side*.csv")
    greeks_files = glob.glob("*volatility-greeks*.csv")
    
    if not side_by_side_files or not greeks_files:
        raise FileNotFoundError("BarchartからダウンロードしたCSVファイル(side-by-side または volatility-greeks)がルートディレクトリに見つかりません。")
        
    # 最新のファイルを採用
    sb_path = sorted(side_by_side_files)[-1]
    gk_path = sorted(greeks_files)[-1]
    
    print(f"[Loading] 建玉データ: {sb_path}")
    print(f"[Loading] ギリシャ・IVデータ: {gk_path}")
    
    # 2. CSVの読み込み (Barchartのヘッダー形式に柔軟に対応)
    df_sb = pd.read_csv(sb_path)
    df_gk = pd.read_csv(gk_path)
    
    # カラム名の前後の空白を削除
    df_sb.columns = [c.strip() for c in df_sb.columns]
    df_gk.columns = [c.strip() for c in df_gk.columns]
    
    # 3. 必要なカラムの抽出と正規化
    # side-by-side から建玉(Open Interest)を取得
    sb_cols = {
        'Strike': 'Strike',
        'Call Open Int': 'Call_OI',
        'Put Open Int': 'Put_OI'
    }
    # Barchartの表記揺れに対応するためのマッピング
    df_sb_renamed = df_sb.rename(columns=lambda x: next((v for k, v in sb_cols.items() if k in x), x))
    
    # volatility-greeks からインプライド・ボラティリティ(IV)を取得
    gk_cols = {
        'Strike': 'Strike',
        'Call Implied Vol': 'Call_IV',
        'Put Implied Vol': 'Put_IV'
    }
    df_gk_renamed = df_gk.rename(columns=lambda x: next((v for k, v in gk_cols.items() if k in x), x))
    
    # 4. Strikeをキーにしてマージ
    df = pd.merge(
        df_sb_renamed[['Strike', 'Call_OI', 'Put_OI']],
        df_gk_renamed[['Strike', 'Call_IV', 'Put_IV']],
        on='Strike', how='inner'
    )
    
    # 文字列型のカンマ除去や数値型への変換
    for col in ['Call_OI', 'Put_OI']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)
            
    for col in ['Call_IV', 'Put_IV']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('%', '').str.replace(',', '').astype(float) / 100.0
            
    df['IV'] = df[['Call_IV', 'Put_IV']].mean(axis=1)
    
    # 限月情報をファイル名から簡易抽出（例：07_28_26）
    expiry_info = "SIU26 (COMEX)"
    if "exp-" in sb_path:
        expiry_info = sb_path.split("exp-")[1].split("-")[0]
        
    return df.sort_values("Strike", ascending=False), expiry_info

def fetch_futures_spot():
    """最新の銀先物(中心限月)のスポット価格を取得"""
    tkr = yf.Ticker("SI=F")
    hist = tkr.history(period="1d")
    if hist.empty:
        raise ValueError("yfinanceから先物スポット価格の取得に失敗しました。")
    return hist['Close'].iloc[-1]

def calculate_gex(df, spot, multiplier=5000):
    """COMEX銀先物のマルチプライヤー（$5,000）を適用した本格GEX計算"""
    df = df[(df['Call_OI'] > 0) | (df['Put_OI'] > 0)].copy()
    T = 22 / 365.0 # 残存日数はCSV記載の22日に固定、あるいは動的計算
    r = 0.045
    iv = np.where(df["IV"] <= 0.01, 0.01, df["IV"])
    
    d1 = (np.log(spot / df["Strike"]) + (r + iv**2 / 2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(T))
    
    # 先物の大きなサイズに合わせてMillion単位に調整
    df["Call_GEX"] = (df["Call_OI"] * gamma * spot * multiplier * 0.01) / 1e6
    df["Put_GEX"] = (-df["Put_OI"] * gamma * spot * multiplier * 0.01) / 1e6
    df["Net_GEX"] = df["Call_GEX"] + df["Put_GEX"]
    return df

def extract_flip_point(df, spot):
    df_sorted = df.sort_values("Strike").dropna(subset=["Net_GEX"])
    net_gex = df_sorted["Net_GEX"].values
    strikes = df_sorted["Strike"].values
    
    sign_flips = np.where(np.diff(np.sign(net_gex)))[0]
    if len(sign_flips) == 0: return np.nan
    
    # 現物価格に最も近い反転ポイントを選択
    closest_flip_idx = min(sign_flips, key=lambda i: abs(strikes[i] - spot))
    x0, x1 = net_gex[closest_flip_idx], net_gex[closest_flip_idx + 1]
    y0, y1 = strikes[closest_flip_idx], strikes[closest_flip_idx + 1]
    
    if x1 - x0 == 0: return y0
    return y0 - (x0 * (y1 - y0) / (x1 - x0))

def export_dashboard(df, spot, expiry, output_path):
    flip_point = extract_flip_point(df, spot)
    
    # 表示範囲を現物価格の上下30%に絞って視認性を極大化
    df_zoom = df[(df['Strike'] >= spot * 0.7) & (df['Strike'] <= spot * 1.3)]
    
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.7, 0.3],
        subplot_titles=("COMEX Silver Futures Dealer Net GEX Profile", "Implied Volatility Smile"),
        shared_xaxes=True, vertical_spacing=0.05
    )
    
    fig.add_trace(go.Bar(x=df_zoom["Strike"], y=df_zoom["Call_GEX"], name="Call GEX", marker_color="rgba(0, 255, 255, 0.6)"), row=1, col=1)
    fig.add_trace(go.Bar(x=df_zoom["Strike"], y=df_zoom["Put_GEX"], name="Put GEX", marker_color="rgba(255, 0, 255, 0.6)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["Net_GEX"], name="Net GEX", mode="lines+markers", line=dict(color="white", width=2)), row=1, col=1)
    
    fig.add_vline(x=spot, line=dict(color="yellow", width=2, dash="solid"), annotation_text=f"Spot (Futures): {spot:.2f}", annotation_position="top left", row=1, col=1)
    if not np.isnan(flip_point):
        fig.add_vline(x=flip_point, line=dict(color="red", width=2, dash="dashdot"), annotation_text=f"Zero-Gamma: {flip_point:.2f}", annotation_position="top right", row=1, col=1)
        
    fig.add_trace(go.Scatter(x=df_zoom["Strike"], y=df_zoom["IV"]*100, name="IV", mode="lines+markers", line_color="orange"), row=2, col=1)
    
    fig.update_layout(
        title=f"Quant Options Radar: 銀先物 (SI) | Expiry: {expiry}",
        template="plotly_dark", height=900, barmode='relative', hovermode='x unified'
    )
    fig.update_yaxes(title_text="GEX ($M)", row=1, col=1)
    fig.update_yaxes(title_text="IV (%)", row=2, col=1)
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)
    
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)

if __name__ == "__main__":
    DOCS_DIR.mkdir(exist_ok=True)

    # 追加: Jekyllビルドを回避するための空ファイルを docs/ 内に強制生成
    (DOCS_DIR / ".nojekyll").touch()
    
    try:
        df, expiry = load_barchart_csv()
        spot = fetch_futures_spot()
        df = calculate_gex(df, spot)
        
        output_path = DOCS_DIR / "slv.html"
        export_dashboard(df, spot, expiry, str(output_path))
        print(f"[SUCCESS] Real Futures GEX Dashboard generated at: {output_path}")
        
    except Exception as e:
        print(f"[ERROR] パイプライン実行失敗: {e}")
