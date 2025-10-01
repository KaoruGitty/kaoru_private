# 株式投資AIシステム

Yahoo!ファイナンスから株価データを取得し、LSTMベースの深層学習モデルで株価予測・天底検出を行い、バックテストで評価する総合的な株式投資AIシステムです。

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
# リポジトリをクローン（または作成）
cd /Volumes/FUKUI-SSD01/fx_trade

# 仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存パッケージをインストール
pip install -r requirements.txt
```

## 📖 使い方

### 基本的な使い方（学習からバックテストまで全て実行）

```bash
python main.py --ticker 7203.T --mode all --epochs 50
```

### 学習のみ実行

```bash
python main.py --ticker 7203.T --mode train --epochs 100 --batch_size 32
```

### バックテストのみ実行（学習済みモデルが必要）

```bash
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
fx_trade/
├── main.py                 # メインスクリプト
├── requirements.txt        # 依存パッケージ
├── README.md              # このファイル
├── data/                  # データと可視化結果
│   ├── processed_data.csv
│   ├── signals_visualization.png
│   ├── *_history.png
│   ├── *_confusion_matrix.png
│   ├── regression_evaluation.png
│   └── backtest_results.png
├── models/                # 学習済みモデル
│   ├── price_regression_best.h5
│   ├── direction_classification_best.h5
│   └── peak_bottom_detection_best.h5
└── src/                   # ソースコード
    ├── data_fetcher.py           # データ取得
    ├── feature_engineering.py    # 特徴量エンジニアリング
    ├── model.py                  # モデル定義
    ├── train.py                  # 学習
    └── backtest.py              # バックテスト・評価
```

## 🔧 各モジュールの説明

### 1. data_fetcher.py
Yahoo!ファイナンスから株価データ（OHLCV）を取得します。

```python
from src.data_fetcher import DataFetcher

fetcher = DataFetcher("7203.T", period="5y")
data = fetcher.fetch_data()
```

### 2. feature_engineering.py
テクニカル指標の計算、天底検出、買いシグナル生成を行います。

主な機能：
- 移動平均線（SMA）: 5, 10, 20, 50, 75, 200日
- 移動平均乖離率
- 指数移動平均（EMA）: 12, 26日
- MACD（Moving Average Convergence Divergence）
- RSI（Relative Strength Index）: 9, 14日
- ストキャスティクス
- ボリンジャーバンド（20日）
- ATR（Average True Range）
- OBV（On Balance Volume）
- 価格変化率、ボラティリティ
- 出来高分析

### 3. model.py
LSTMベースの深層学習モデルを定義します。

- **回帰モデル**: 将来の価格変化率を予測
- **分類モデル**: 価格の上昇/横ばい/下降を予測
- **天底検出モデル**: ピーク（天井）、ボトム（底）、通常を分類

### 4. train.py
モデルの学習を統合的に管理します。

### 5. backtest.py
学習済みモデルの評価とバックテストを実行します。

## 📊 出力結果

### 可視化ファイル（data/ディレクトリ）

1. **signals_visualization.png**: 株価チャートと買いシグナル、天底の可視化
2. **price_regression_history.png**: 回帰モデルの学習履歴
3. **direction_classification_history.png**: 分類モデルの学習履歴
4. **peak_bottom_detection_history.png**: 天底検出モデルの学習履歴
5. **regression_evaluation.png**: 回帰モデルの予測精度
6. **direction_classification_confusion_matrix.png**: 方向予測の混同行列
7. **peak_bottom_detection_confusion_matrix.png**: 天底検出の混同行列
8. **backtest_results.png**: バックテスト結果（売買シグナルとポートフォリオ価値）

### モデルファイル（models/ディレクトリ）

- `price_regression_best.h5`: 価格予測モデル
- `direction_classification_best.h5`: 方向予測モデル
- `peak_bottom_detection_best.h5`: 天底検出モデル

## 🎯 使用例

### 例1: トヨタ自動車で学習とバックテスト

```bash
python main.py --ticker 7203.T --mode all --epochs 50
```

### 例2: ソフトバンクグループで詳細な学習

```bash
python main.py --ticker 9984.T --mode train --epochs 100 --lookback 90
```

### 例3: 既存モデルでバックテストのみ

```bash
python main.py --ticker 7203.T --mode backtest --initial_capital 5000000
```

## 📈 バックテスト戦略

現在実装されているトレーディング戦略：

- **買いシグナル**: 価格が上昇すると予測され（direction=2）、かつ予測リターンが1%以上
- **売りシグナル**: 価格が下降すると予測され（direction=0）、または予測リターンが-0.5%未満

この戦略はカスタマイズ可能です（`src/backtest.py`の`run_backtest`メソッドを編集）。

## ⚠️ 注意事項

1. **投資は自己責任で**: このシステムは教育目的であり、実際の投資判断は自己責任で行ってください
2. **過去の成績は将来を保証しない**: バックテストの結果が良くても、実際の市場では異なる結果になる可能性があります
3. **データの精度**: Yahoo!ファイナンスのデータに依存しています
4. **計算リソース**: LSTMモデルの学習には時間がかかる場合があります（GPU推奨）

## 🔄 今後の改善案

- [ ] より高度なテクニカル指標の追加
- [ ] アンサンブル学習の実装
- [ ] リアルタイムトレーディング機能
- [ ] 複数銘柄の同時分析
- [ ] ハイパーパラメータの自動最適化
- [ ] Transformer モデルの実装
- [ ] 感情分析（ニュース、SNS）の統合

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 貢献

プルリクエストを歓迎します！大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 📧 お問い合わせ

質問や提案がある場合は、GitHubのissueを開いてください。

---

**免責事項**: このソフトウェアは「現状のまま」提供され、明示的または黙示的な保証はありません。このソフトウェアの使用により生じたいかなる損害についても、作者は責任を負いません。

