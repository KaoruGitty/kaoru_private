import pandas as pd
import numpy as np
import os
from src.data_fetcher import DataFetcher
from src.feature_engineering import FeatureEngineering
from src.model import LSTMModel, TimeSeriesDataPreparator
from src.backtest import Backtester
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

def main():
    ticker = "7203.T"
    lookback = 60
    test_size = 0.2
    
    print("=" * 80)
    print("Focal Loss + 時系列CV 改善版モデル バックテスト実行")
    print("=" * 80)
    
    # データの読み込みと準備
    print("\n📊 改善版データを読み込み中...")
    data_fetcher = DataFetcher(ticker, period="10y")  # 10年分のデータを取得
    data = data_fetcher.fetch_data()
    print(f"✅ データ読み込み完了: {len(data)}行")
    print(f"   期間: {data.index[0]} ～ {data.index[-1]}")
    
    feature_engineer = FeatureEngineering(data.copy())
    feature_engineer.add_technical_indicators()
    feature_engineer.detect_peaks_and_bottoms()
    feature_engineer.create_target_labels()
    feature_engineer.create_time_series_features()
    data = feature_engineer.handle_missing_values()
    
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
    
    print(f"✅ 特徴量選択完了: {len(feature_columns)}個")
    
    # バックテスターを初期化
    print("\n🤖 バックテスターを初期化中...")
    backtester = Backtester(
        data=data,
        feature_columns=feature_columns,
        lookback=lookback
    )
    
    # 改善版モデルを読み込む
    print("\n📂 改善版モデルを読み込み中...")
    try:
        backtester.load_models(models_dir="/Volumes/FUKUI-SSD01/fx_trade/models")
        print("✅ モデル読み込み完了")
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return
    
    # 予測を実行
    print("\n🚀 予測を実行中...")
    try:
        # 評価データセットの準備
        required_cols = feature_columns + ['target_return', 'target_class', 'target_peak_bottom']
        df_clean = data[required_cols].dropna()
        
        total_sequences = len(df_clean) - lookback
        split_idx = int(total_sequences * (1 - test_size))
        
        X_data = df_clean[feature_columns].values.astype(np.float32)
        y_data_reg = df_clean['target_return'].values.astype(np.float32)
        y_data_class = df_clean['target_class'].values.astype(np.float32)
        y_data_peak_bottom = df_clean['target_peak_bottom'].values.astype(np.float32)
        
        data_preparator = TimeSeriesDataPreparator(lookback=lookback)
        X, y_reg_seq = data_preparator.create_sequences(X_data, y_data_reg)
        _, y_class_seq = data_preparator.create_sequences(X_data, y_data_class)
        _, y_peak_bottom_seq = data_preparator.create_sequences(X_data, y_data_peak_bottom)
        
        X_test = X[split_idx:]
        y_test_reg = y_reg_seq[split_idx:]
        y_test_class = y_class_seq[split_idx:]
        y_test_peak_bottom = y_peak_bottom_seq[split_idx:]
        
        print(f"テストデータ: {len(X_test)}行")
        print(f"X_test shape: {X_test.shape}")
        
        print("全モデルで予測実行中...")
        backtester.predictions['regression'] = {'y_pred': backtester.regression_model.predict(X_test)}
        print(f"回帰予測完了: {len(backtester.predictions['regression']['y_pred'])}個")
        
        # 分類モデルの予測 (方向)
        y_pred_class_probs = backtester.classification_model.predict(X_test)
        y_pred_class = np.argmax(y_pred_class_probs, axis=1)
        backtester.predictions['classification'] = {'y_pred': y_pred_class}
        print(f"分類予測完了: {len(backtester.predictions['classification']['y_pred'])}個")
        
        # 天底検出モデルの予測
        y_pred_peak_bottom_probs = backtester.peak_bottom_model.predict(X_test)
        y_pred_peak_bottom = np.argmax(y_pred_peak_bottom_probs, axis=1)
        backtester.predictions['peak_bottom'] = {'y_pred': y_pred_peak_bottom}
        print(f"天底予測完了: {len(backtester.predictions['peak_bottom']['y_pred'])}個")
        
    except Exception as e:
        print(f"❌ 予測実行エラー: {e}")
        return
    
    # バックテストを実行
    print("\n🚀 バックテストを実行中...")
    try:
        results = backtester.run_backtest(initial_capital=1_000_000, test_size=test_size)
        
        print("\n" + "=" * 80)
        print("📈 バックテスト結果")
        print("=" * 80)
        if results:
            print(f"💰 初期資本: ¥{results['initial_capital']:,}")
            print(f"📈 最終資本: ¥{results['final_capital']:,}")
            print(f"📊 総リターン: {results['total_return']:.2f}%")
            print(f"📉 Buy & Hold リターン: {results['buy_and_hold_return']:.2f}%")
            print(f"🚀 優位性: {results['outperformance']:.2f}%")
            print(f"🔄 総取引数: {results['total_trades']}回")
            print(f"⬇️ 最大ドローダウン: {results['max_drawdown']:.2f}%")
            print(f"✅ 勝率: {results['win_rate']:.2f}%")
            print(f"平均利益: {results['average_profit']:.2f}")
            print(f"平均損失: {results['average_loss']:.2f}")
            print(f"プロフィットファクター: {results['profit_factor']:.2f}")
            print(f"シャープ・レシオ: {results['sharpe_ratio']:.2f}")
        else:
            print("❌ バックテスト実行エラー")
    except Exception as e:
        print(f"❌ バックテスト実行エラー: {e}")
    
    # 予測結果の分析
    print("\n=== 予測結果の分析 ===")
    print("回帰予測の統計:")
    print(f"  平均: {np.mean(backtester.predictions['regression']['y_pred']):.4f}")
    print(f"  標準偏差: {np.std(backtester.predictions['regression']['y_pred']):.4f}")
    print(f"  最小値: {np.min(backtester.predictions['regression']['y_pred']):.4f}")
    print(f"  最大値: {np.max(backtester.predictions['regression']['y_pred']):.4f}")
    
    print("\n分類予測（方向）の分布:")
    unique_classes, counts = np.unique(backtester.predictions['classification']['y_pred'], return_counts=True)
    class_map = {0: "下降", 1: "横ばい", 2: "上昇"}
    for cls, count in zip(unique_classes, counts):
        print(f"  {class_map.get(cls, '不明')}: {count}回 ({count / len(backtester.predictions['classification']['y_pred']) * 100:.1f}%)")
    
    print("\n天底予測の分布:")
    unique_classes_pb, counts_pb = np.unique(backtester.predictions['peak_bottom']['y_pred'], return_counts=True)
    class_map_pb = {0: "通常", 1: "天井", 2: "底"}
    for cls, count in zip(unique_classes_pb, counts_pb):
        print(f"  {class_map_pb.get(cls, '不明')}: {count}回 ({count / len(backtester.predictions['peak_bottom']['y_pred']) * 100:.1f}%)")
    
    # 評価指標の計算と表示
    print("\n=== モデル評価指標 ===")
    try:
        # 回帰モデルの評価
        y_true_reg = y_test_reg
        y_pred_reg = backtester.predictions['regression']['y_pred']
        
        mae = mean_absolute_error(y_true_reg, y_pred_reg)
        rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
        r2 = r2_score(y_true_reg, y_pred_reg)
        
        print(f"回帰モデル (価格変動率予測):")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2 Score: {r2:.4f}")
        
        # 分類モデル（方向予測）の評価
        y_true_class = y_test_class
        y_pred_class = backtester.predictions['classification']['y_pred']
        
        print("\n分類モデル (価格方向予測):")
        print(classification_report(y_true_class, y_pred_class, target_names=["下降", "横ばい", "上昇"], zero_division=0))
        
        # 天底検出モデルの評価
        y_true_peak_bottom = y_test_peak_bottom
        y_pred_peak_bottom = backtester.predictions['peak_bottom']['y_pred']
        
        print("\n分類モデル (天底検出):")
        print(classification_report(y_true_peak_bottom, y_pred_peak_bottom, target_names=["通常", "天井", "底"], zero_division=0))
        
    except Exception as e:
        print(f"❌ 評価指標計算エラー: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 Focal Loss + 時系列CV 改善版バックテスト完了！")
    print("=" * 80)

if __name__ == "__main__":
    main()
