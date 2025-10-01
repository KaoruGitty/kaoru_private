"""
株式投資AIシステム - メインスクリプト

Yahoo!ファイナンスから株価データを取得し、テクニカル指標を計算、
LSTMモデルで学習・予測、バックテストを実行
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import ModelTrainer
from src.backtest import Backtester
import argparse


def main():
    """
    メイン処理
    """
    parser = argparse.ArgumentParser(description='株式投資AIシステム')
    parser.add_argument('--ticker', type=str, default='7203.T', help='銘柄コード（例: 7203.T for トヨタ自動車）')
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'backtest', 'all'], help='実行モード')
    parser.add_argument('--lookback', type=int, default=60, help='過去何日分のデータを使用するか')
    parser.add_argument('--test_size', type=float, default=0.2, help='テストデータの割合')
    parser.add_argument('--epochs', type=int, default=50, help='学習エポック数')
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--initial_capital', type=float, default=1000000, help='バックテストの初期資本')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("株式投資AIシステム")
    print("=" * 80)
    print(f"銘柄: {args.ticker}")
    print(f"モード: {args.mode}")
    print(f"Lookback期間: {args.lookback}日")
    print(f"テストデータ割合: {args.test_size * 100}%")
    print("=" * 80 + "\n")
    
    if args.mode in ['train', 'all']:
        # 学習フェーズ
        print("\n" + "=" * 80)
        print("学習フェーズ開始")
        print("=" * 80)
        
        trainer = ModelTrainer(
            ticker=args.ticker,
            lookback=args.lookback,
            test_size=args.test_size
        )
        
        trainer.train_all_models(
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print("\n学習が完了しました！")
        print("モデルは models/ ディレクトリに保存されています。")
        print("可視化結果は data/ ディレクトリに保存されています。")
    
    if args.mode in ['backtest', 'all']:
        # バックテストフェーズ
        print("\n" + "=" * 80)
        print("バックテスト・評価フェーズ開始")
        print("=" * 80)
        
        # データを読み込む
        import pandas as pd
        data_path = '/Volumes/FUKUI-SSD01/fx_trade/data/processed_data.csv'
        
        if not os.path.exists(data_path):
            print(f"処理済みデータが見つかりません: {data_path}")
            print("まず学習を実行してください (--mode train)")
            return
        
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # 特徴量を選択（学習時と同じ）
        base_features = ['open', 'high', 'low', 'close', 'volume']
        technical_features = [col for col in data.columns if any(
            indicator in col for indicator in [
                'sma', 'ema', 'macd', 'rsi', 'stoch', 'bb', 'atr', 'obv',
                'return', 'volatility', 'volume_sma', 'volume_ratio'
            ]
        )]
        time_features = ['day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter']
        feature_columns = base_features + technical_features + time_features
        feature_columns = [col for col in feature_columns if col in data.columns]
        
        # ターゲット変数や不要な列を除外
        exclude_columns = ['next_day_return', 'target_return', 'target_class', 'target_class_mapped',
                          'target_peak_bottom', 'is_peak', 'is_bottom', 'buy_signal']
        feature_columns = [col for col in feature_columns if col not in exclude_columns]
        
        # バックテスターを初期化
        backtester = Backtester(
            data=data,
            feature_columns=feature_columns,
            lookback=args.lookback
        )
        
        # モデルを読み込む
        backtester.load_models()
        
        # モデルを評価
        backtester.evaluate_regression_model(test_size=args.test_size)
        backtester.evaluate_classification_model(test_size=args.test_size)
        backtester.evaluate_peak_bottom_model(test_size=args.test_size)
        
        # バックテストを実行
        results = backtester.run_backtest(
            initial_capital=args.initial_capital,
            test_size=args.test_size
        )
        
        print("\nバックテスト・評価が完了しました！")
        print("結果は data/ ディレクトリに保存されています。")
    
    print("\n" + "=" * 80)
    print("全処理が完了しました！")
    print("=" * 80)


if __name__ == "__main__":
    main()

