# Options GEX Radar (オプション・ガンマ・エクスポージャー・レーダー)

## Architecture Overview
本システムは、オプション市場におけるマーケットメーカー（MM）のガンマ・エクスポージャー（Net GEX）を定量化し、原資産（CFD/先物）の価格推移における「重力場（Gamma Pinning）」および「非線形なスクイーズ（Gamma Squeeze）」の閾値を動的に探知する静的クオンツパイプラインである。

## Core Components
- **Ingestion**: ヘッドレスブラウザ(Playwright)によるEOD(日次)オプションチェーンの取得
- **Compute**: Vectorized Black-Scholesエンジンによる $\Gamma$ およびNet GEXの高速算出
- **Presentation**: Plotlyを用いたモバイルレスポンシブな静的HTMLダッシュボードのエクスポートとGitHub PagesへのCI/CDデプロイ
