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

class FlexibleBacktester(Backtester):
    """
    より柔軟な取引戦略を持つバックテスター
    """
    
    def run_flexible_backtest(self, initial_capital: float = 1000000, test_size: float = 0.2):
        """
        柔軟な取引戦略でバックテストを実行
        """
        print("\n" + "=" * 80)
        print("Running Flexible Backtest")
        print("=" * 80)
        
        # データをクリーニング（評価時と同じ方法）
        required_cols = self.feature_columns + ['target_return']
        df_clean = self.data[required_cols].dropna()
        
        # 時系列データなので単純分割（最新データをテストに）
        total_sequences = len(df_clean) - self.lookback
        split_idx = int(total_sequences * (1 - test_size))
        
        # テストデータの実際のインデックスを取得
        test_start_idx = split_idx + self.lookback
        test_end_idx = test_start_idx + len(self.predictions['regression']['y_pred'])
        
        # 元のデータフレームからテストデータを取得
        test_data = df_clean.iloc[test_start_idx:test_end_idx].copy()
        
        # 予測結果を追加
        test_data['predicted_return'] = self.predictions['regression']['y_pred']
        test_data['predicted_direction'] = self.predictions['classification']['y_pred']
        test_data['predicted_peak_bottom'] = self.predictions['peak_bottom']['y_pred']
        
        # closeカラムを元のデータから取得
        if 'close' not in test_data.columns:
            test_data = test_data.join(self.data[['close']], how='left')
        
        print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
        print(f"Test data length: {len(test_data)}")
        
        # 柔軟なトレーディング戦略
        capital = initial_capital
        position = 0  # 0: なし, 1: 買いポジション
        shares = 0
        trades = []
        portfolio_values = [initial_capital]
        
        # 取引統計
        buy_signals = 0
        sell_signals = 0
        
        for i in range(len(test_data)):
            row = test_data.iloc[i]
            current_price = row['close']
            
            # 柔軟な買いシグナル
            buy_conditions = [
                # 条件1: 上昇予測 + 正のリターン予測
                (row['predicted_direction'] == 2 and row['predicted_return'] > 0.005),
                # 条件2: 底値予測 + 正のリターン予測
                (row['predicted_peak_bottom'] == 2 and row['predicted_return'] > 0.003),
                # 条件3: 横ばいでも強い正のリターン予測
                (row['predicted_direction'] == 1 and row['predicted_return'] > 0.015),
                # 条件4: 通常でも中程度の正のリターン予測
                (row['predicted_peak_bottom'] == 0 and row['predicted_return'] > 0.008)
            ]
            
            if position == 0 and any(buy_conditions):
                # 買い
                shares = capital / current_price
                position = 1
                buy_signals += 1
                trades.append({
                    'date': row.name,
                    'action': 'buy',
                    'price': current_price,
                    'shares': shares,
                    'capital': capital,
                    'predicted_return': row['predicted_return'],
                    'predicted_direction': row['predicted_direction'],
                    'predicted_peak_bottom': row['predicted_peak_bottom']
                })
                print(f"BUY  {row.name.date()}: Price={current_price:.2f}, PredReturn={row['predicted_return']:.4f}, Direction={row['predicted_direction']}, PeakBottom={row['predicted_peak_bottom']}")
            
            # 柔軟な売りシグナル
            sell_conditions = [
                # 条件1: 下降予測
                (row['predicted_direction'] == 0),
                # 条件2: 天井予測
                (row['predicted_peak_bottom'] == 1),
                # 条件3: 負のリターン予測
                (row['predicted_return'] < -0.008),
                # 条件4: 横ばい + 負のリターン予測
                (row['predicted_direction'] == 1 and row['predicted_return'] < -0.003)
            ]
            
            if position == 1 and any(sell_conditions):
                # 売り
                capital = shares * current_price
                sell_signals += 1
                trades.append({
                    'date': row.name,
                    'action': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'capital': capital,
                    'predicted_return': row['predicted_return'],
                    'predicted_direction': row['predicted_direction'],
                    'predicted_peak_bottom': row['predicted_peak_bottom']
                })
                print(f"SELL {row.name.date()}: Price={current_price:.2f}, Capital={capital:.2f}, PredReturn={row['predicted_return']:.4f}")
                shares = 0
                position = 0
            
            # ポートフォリオ価値を計算
            if position == 1:
                portfolio_value = shares * current_price
            else:
                portfolio_value = capital
            
            portfolio_values.append(portfolio_value)
        
        # 最終的にポジションを持っていれば清算
        if position == 1:
            final_price = test_data.iloc[-1]['close']
            capital = shares * final_price
            trades.append({
                'date': test_data.index[-1],
                'action': 'sell',
                'price': final_price,
                'shares': shares,
                'capital': capital,
                'predicted_return': test_data.iloc[-1]['predicted_return'],
                'predicted_direction': test_data.iloc[-1]['predicted_direction'],
                'predicted_peak_bottom': test_data.iloc[-1]['predicted_peak_bottom']
            })
            print(f"FINAL SELL: Price={final_price:.2f}, Capital={capital:.2f}")
        
        # 結果を計算
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        # Buy & Hold リターン
        buy_hold_return = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
        
        # その他の指標
        total_trades = len([t for t in trades if t['action'] == 'buy'])
        
        # ドローダウン計算
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        max_drawdown = np.min(drawdown)
        
        # 勝率計算
        if len(trades) >= 2:
            profits = []
            for i in range(1, len(trades), 2):
                if i < len(trades):
                    profit = trades[i]['capital'] - trades[i-1]['capital']
                    profits.append(profit)
            
            if profits:
                wins = len([p for p in profits if p > 0])
                win_rate = wins / len(profits) * 100
                average_profit = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
                average_loss = np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0
                profit_factor = abs(sum([p for p in profits if p > 0]) / sum([p for p in profits if p < 0])) if any(p < 0 for p in profits) else float('inf')
            else:
                win_rate = 0
                average_profit = 0
                average_loss = 0
                profit_factor = 0
        else:
            win_rate = 0
            average_profit = 0
            average_loss = 0
            profit_factor = 0
        
        # シャープレシオ計算
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'buy_and_hold_return': buy_hold_return,
            'outperformance': total_return - buy_hold_return,
            'total_trades': total_trades,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'average_profit': average_profit,
            'average_loss': average_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
        
        print("\n" + "=" * 80)
        print("Flexible Backtest Results")
        print("=" * 80)
        print(f"Initial Capital: ¥{initial_capital:,}")
        print(f"Final Capital: ¥{final_capital:,}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"Outperformance: {total_return - buy_hold_return:.2f}%")
        print(f"Number of Trades: {total_trades}")
        print(f"Buy Signals: {buy_signals}")
        print(f"Sell Signals: {sell_signals}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Profit: {average_profit:.2f}")
        print(f"Average Loss: {average_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        return results

def main():
    ticker = "7203.T"
    lookback = 60
    test_size = 0.2
    
    print("=" * 80)
    print("柔軟な取引戦略 バックテスト実行")
    print("=" * 80)
    
    # データの読み込みと準備
    print("\n📊 データを読み込み中...")
    data_fetcher = DataFetcher(ticker, period="10y")
    data = data_fetcher.fetch_data()
    print(f"✅ データ読み込み完了: {len(data)}行")
    
    feature_engineer = FeatureEngineering(data.copy())
    feature_engineer.add_technical_indicators()
    feature_engineer.detect_peaks_and_bottoms()
    feature_engineer.create_target_labels()
    feature_engineer.create_time_series_features()
    data = feature_engineer.handle_missing_values()
    
    # 特徴量を選択
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
    
    # 柔軟なバックテスターを初期化
    print("\n🤖 柔軟なバックテスターを初期化中...")
    backtester = FlexibleBacktester(
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
    
    # 柔軟なバックテストを実行
    print("\n🚀 柔軟なバックテストを実行中...")
    try:
        results = backtester.run_flexible_backtest(initial_capital=1_000_000, test_size=test_size)
        
        print("\n" + "=" * 80)
        print("📈 柔軟なバックテスト結果")
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
    
    print("\n" + "=" * 80)
    print("🎉 柔軟な取引戦略バックテスト完了！")
    print("=" * 80)

if __name__ == "__main__":
    main()
