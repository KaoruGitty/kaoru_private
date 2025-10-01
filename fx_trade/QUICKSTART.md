# クイックスタートガイド

このガイドでは、株式投資AIシステムをすぐに使い始める方法を説明します。

## 📦 1. セットアップ（初回のみ）

```bash
# プロジェクトディレクトリに移動
cd /Volumes/FUKUI-SSD01/fx_trade

# 仮想環境を作成
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate

# 必要なパッケージをインストール
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 2. 基本的な使い方

### シンプルな実行（すべて自動）

```bash
# トヨタ自動車（7203.T）で学習からバックテストまで一括実行
python main.py --ticker 7203.T --mode all --epochs 50
```

実行すると以下が自動で行われます：
1. Yahoo!ファイナンスからデータ取得（過去5年分）
2. テクニカル指標の計算（移動平均、RSI、MACD、ボリンジャーバンドなど）
3. 天底ポイントと買いシグナルの検出
4. 3つのLSTMモデルの学習
   - 価格変化率予測（回帰）
   - 価格変動方向予測（分類）
   - 天底検出（分類）
5. モデルの評価
6. バックテストの実行
7. 結果の可視化

### 実行時間の目安

- CPU: 約30-60分
- GPU: 約10-20分

## 📊 3. 結果の確認

### 生成されるファイル

#### data/ディレクトリ
```
data/
├── processed_data.csv                          # 処理済みデータ
├── signals_visualization.png                   # 買いシグナルと天底の可視化
├── price_regression_history.png                # 回帰モデル学習履歴
├── direction_classification_history.png        # 分類モデル学習履歴
├── peak_bottom_detection_history.png          # 天底検出モデル学習履歴
├── regression_evaluation.png                   # 予測精度評価
├── direction_classification_confusion_matrix.png
├── peak_bottom_detection_confusion_matrix.png
└── backtest_results.png                        # バックテスト結果
```

#### models/ディレクトリ
```
models/
├── price_regression_best.h5          # 価格予測モデル
├── direction_classification_best.h5  # 方向予測モデル
└── peak_bottom_detection_best.h5     # 天底検出モデル
```

### ターミナル出力の例

```
================================================================================
学習フェーズ開始
================================================================================
Fetching data for 7203.T...
Data fetched: 1258 rows from 2019-01-04 to 2024-01-03

Technical indicators added. Total features: 87
Peaks detected: 45
Bottoms detected: 42
Buy signals generated: 18

================================================================================
バックテスト結果
================================================================================
Initial Capital: ¥1,000,000
Final Capital: ¥1,245,320
Total Return: 24.53%
Buy & Hold Return: 18.25%
Outperformance: 6.28%
Number of Trades: 12
```

## 🎯 4. 他の銘柄で試す

### 人気の日本株

```bash
# ソフトバンクグループ
python main.py --ticker 9984.T --mode all --epochs 50

# ソニーグループ
python main.py --ticker 6758.T --mode all --epochs 50

# 任天堂
python main.py --ticker 7974.T --mode all --epochs 50

# ファーストリテイリング（ユニクロ）
python main.py --ticker 9983.T --mode all --epochs 50

# キーエンス
python main.py --ticker 6861.T --mode all --epochs 50
```

### 米国株

```bash
# Apple
python main.py --ticker AAPL --mode all --epochs 50

# Microsoft
python main.py --ticker MSFT --mode all --epochs 50

# Tesla
python main.py --ticker TSLA --mode all --epochs 50
```

## ⚙️ 5. 詳細なオプション

### 学習のみ実行（より多くのエポック）

```bash
python main.py --ticker 7203.T --mode train --epochs 100 --batch_size 64
```

### バックテストのみ実行（既存モデル使用）

```bash
python main.py --ticker 7203.T --mode backtest --initial_capital 5000000
```

### より長い履歴で学習

```bash
python main.py --ticker 7203.T --mode all --lookback 90 --epochs 50
```

## 🔍 6. トラブルシューティング

### エラー: "No data fetched"

銘柄コードが正しいか確認してください。日本株の場合は `.T` を付けます。
```bash
# 正しい例
python main.py --ticker 7203.T

# 間違い
python main.py --ticker 7203
```

### エラー: "Model not found"

バックテストを実行する前に学習を実行してください。
```bash
python main.py --ticker 7203.T --mode train --epochs 50
```

### メモリエラー

バッチサイズを小さくしてみてください。
```bash
python main.py --ticker 7203.T --mode all --batch_size 16
```

### GPU が使えない

TensorFlowが自動的にCPUを使用します。GPUがある場合：
```bash
# GPUが認識されているか確認
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 📈 7. 結果の見方

### signals_visualization.png
- 緑の三角（▲）: 検出された底値
- 赤の三角（▼）: 検出された天井
- 青い星（★）: 買いシグナル（底値 + 翌日上昇）

### backtest_results.png
- 上のグラフ: 実際の株価と売買ポイント
  - 緑の三角: 買い注文
  - 赤の三角: 売り注文
- 下のグラフ: ポートフォリオ価値の推移

### 評価指標の意味
- **Total Return**: 総リターン（初期資本からの増減率）
- **Buy & Hold Return**: 単純に買って保持した場合のリターン
- **Outperformance**: AIシステムの超過リターン
- **MAE**: 平均絶対誤差（小さいほど予測精度が高い）
- **RMSE**: 二乗平均平方根誤差
- **R2 Score**: 決定係数（1に近いほど良い）
- **Accuracy**: 分類精度

## 🎓 8. 次のステップ

### カスタマイズ

1. **トレーディング戦略を変更**: `src/backtest.py` の `run_backtest` メソッドを編集
2. **テクニカル指標を追加**: `src/feature_engineering.py` の `add_technical_indicators` メソッドを編集
3. **モデル構造を変更**: `src/model.py` のモデル構築メソッドを編集

### より高度な使い方

Pythonスクリプトから直接使用：

```python
from src.train import ModelTrainer
from src.backtest import Backtester

# 学習
trainer = ModelTrainer("7203.T", lookback=60)
trainer.train_all_models(epochs=50)

# バックテスト
backtester = Backtester(trainer.data, trainer.feature_columns)
backtester.load_models()
backtester.evaluate_regression_model()
results = backtester.run_backtest(initial_capital=1000000)
```

## ⚠️ 9. 重要な注意事項

1. **投資は自己責任**: このシステムは教育・研究目的です
2. **過去の成績は将来を保証しない**: バックテストの好成績が実際の利益を保証するものではありません
3. **リスク管理**: 実際の投資では必ず損切りルールを設定してください
4. **分散投資**: 1銘柄に集中せず、複数銘柄に分散してください

## 📚 10. さらに学ぶ

- 詳細なドキュメント: `README.md`
- ソースコード: `src/` ディレクトリ
- テクニカル分析の基礎: 移動平均線、RSI、MACD などを学ぶ
- 機械学習の基礎: LSTM、時系列予測について学ぶ

---

質問や問題があれば、GitHubのissueを開いてください。

Happy Trading! 📈💰

