# AI Trading Agent System

Yahoo!ファイナンスから株価データを取得し、LSTMベースの深層学習モデルで株価予測・天底検出を行い、バックテストで評価する総合的な株式投資AIシステムです。

## 🤖 AIエージェント機能

- **IntelligentTradingAgent**: メインAIエージェント
- **MarketRegimeDetector**: 市場環境検出（強気・弱気・横ばい・高ボラティリティ）
- **AdaptiveThresholdAgent**: 動的閾値調整
- **MultiModelEnsembleAgent**: LSTM + Random Forest統合
- **RiskManagementAgent**: 高度なリスク管理

## 🌟 主な機能

1. **データ取得**: Yahoo!ファイナンスから過去の株価データを取得
2. **テクニカル指標**: 移動平均乖離率、ボリンジャーバンド、RSI、MACD、ATR、OBV等を自動計算
3. **天底検出**: ローカルピーク・ボトムを自動検出し、買いシグナルを生成
4. **特徴量エンジニアリング**: 時系列特徴量の作成、欠損値補完、正規化
5. **LSTMモデル**: 
   - 価格変化率予測（回帰）
   - 価格変動方向予測（分類）
   - 天底検出（分類）
6. **バックテスト**: 実際のトレーディング戦略をシミュレーション
7. **可視化**: 全ての結果を分かりやすくグラフ化

## 📋 必要要件

- Python 3.8以上
- 必要なライブラリは `requirements.txt` に記載

## 🚀 インストール

```bash
# リポジトリをクローン
git clone https://github.com/KaoruGitty/kaoru_private.git
cd kaoru_private

# 仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存パッケージをインストール
pip install -r requirements.txt
```

## 📖 使い方

### AIエージェントシステム

```bash
# fx_tradeディレクトリに移動
cd fx_trade

# AIエージェントの初期化とバックテスト
python ai_trading_agent.py
python ai_agent_backtest.py
```

### 基本的な使い方（学習からバックテストまで全て実行）

```bash
# fx_tradeディレクトリに移動
cd fx_trade

python main.py --ticker 7203.T --mode all --epochs 50
```

### 学習のみ実行

```bash
# fx_tradeディレクトリに移動
cd fx_trade

python main.py --ticker 7203.T --mode train --epochs 100 --batch_size 32
```

### バックテストのみ実行（学習済みモデルが必要）

```bash
# fx_tradeディレクトリに移動
cd fx_trade

python main.py --ticker 7203.T --mode backtest --initial_capital 1000000
```

### コマンドライン引数

- `--ticker`: 銘柄コード（例: `7203.T` トヨタ自動車、`9984.T` ソフトバンクグループ）
- `--mode`: 実行モード (`train`, `backtest`, `all`)
- `--lookback`: 過去何日分のデータを使用するか（デフォルト: 60日）
- `--test_size`: テストデータの割合（デフォルト: 0.2 = 20%）
- `--epochs`: 学習エポック数（デフォルト: 50）
- `--batch_size`: バッチサイズ（デフォルト: 32）
- `--initial_capital`: バックテストの初期資本（デフォルト: 1,000,000円）

## 📁 プロジェクト構造

```
kaoru_private/
├── fx_trade/                  # AI Trading Agent System
│   ├── ai_trading_agent.py        # AIエージェントシステム
│   ├── ai_agent_backtest.py       # AIエージェントバックテスト
│   ├── main.py                    # メインスクリプト
│   ├── requirements.txt           # 依存パッケージ
│   ├── README.md                 # このファイル
│   ├── data/                     # データと可視化結果
│   │   ├── processed_data.csv
│   │   ├── signals_visualization.png
│   │   └── ai_agent_backtest_results.png
│   ├── models/                   # 学習済みモデル
│   │   ├── *_focal_best.h5
│   │   └── *_focal.h5
│   └── src/                      # ソースコード
│       ├── data_fetcher.py           # データ取得
│       ├── feature_engineering.py    # 特徴量エンジニアリング
│       ├── model.py                  # モデル定義
│       ├── train.py                  # 学習
│       ├── backtest.py              # バックテスト・評価
│       ├── focal_loss.py            # Focal Loss実装
│       └── time_series_cv.py        # 時系列クロスバリデーション
├── *.ipynb                    # 他のMLプロジェクト
└── *.py                       # 他のMLスクリプト
```

## 🔧 各モジュールの説明

### 1. ai_trading_agent.py
高度なAIエージェントシステム：
- 市場環境の自動検出
- 適応的閾値調整
- マルチモデルアンサンブル
- 高度なリスク管理

### 2. data_fetcher.py
Yahoo!ファイナンスから株価データ（OHLCV）を取得します。

### 3. feature_engineering.py
テクニカル指標の計算、天底検出、買いシグナル生成を行います。

### 4. model.py
LSTMベースの深層学習モデルを定義します。

### 5. train.py / train_focal_tscv.py
モデルの学習を統合的に管理します。

### 6. backtest.py
学習済みモデルの評価とバックテストを実行します。

## 📊 出力結果

### AIエージェントバックテスト結果
- ポートフォリオ価値の推移
- 取引シグナルの可視化
- パフォーマンス指標
- ドローダウン分析

### 従来システムの結果
- 学習履歴グラフ
- 混同行列
- 回帰評価
- バックテスト結果

## 🎯 使用例

### AIエージェントシステム

```bash
# fx_tradeディレクトリに移動
cd fx_trade

# AIエージェントの実行
python ai_trading_agent.py
python ai_agent_backtest.py
```

### 従来システム

```bash
# fx_tradeディレクトリに移動
cd fx_trade

# トヨタ自動車で学習とバックテスト
python main.py --ticker 7203.T --mode all --epochs 50

# Focal Loss + 時系列CVで学習
python train_focal_tscv.py
```

## 📈 バックテスト戦略

### AIエージェント戦略
- 市場環境に応じた動的調整
- 信頼度ベースのポジションサイズ決定
- 複数モデルのアンサンブル予測
- 高度なリスク管理

### 従来戦略
- 固定閾値による買いシグナル
- 価格方向予測ベースの取引

## ⚠️ 注意事項

1. **投資は自己責任で**: このシステムは教育目的であり、実際の投資判断は自己責任で行ってください
2. **過去の成績は将来を保証しない**: バックテストの結果が良くても、実際の市場では異なる結果になる可能性があります
3. **データの精度**: Yahoo!ファイナンスのデータに依存しています
4. **計算リソース**: LSTMモデルの学習には時間がかかる場合があります（GPU推奨）

## 🔄 今後の改善案

- [x] AIエージェントシステムの実装
- [x] 市場環境検出機能
- [x] 適応的閾値調整
- [x] マルチモデルアンサンブル
- [x] Focal Loss実装
- [x] 時系列クロスバリデーション
- [ ] 強化学習の導入
- [ ] リアルタイムトレーディング機能
- [ ] 複数銘柄の同時分析
- [ ] 感情分析（ニュース、SNS）の統合

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 貢献

プルリクエストを歓迎します！大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 📧 お問い合わせ

質問や提案がある場合は、GitHubのissueを開いてください。

---

**免責事項**: このソフトウェアは「現状のまま」提供され、明示的または黙示的な保証はありません。このソフトウェアの使用により生じたいかなる損害についても、作者は責任を負いません。
