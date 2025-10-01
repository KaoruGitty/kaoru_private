"""
バックテストと評価
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from .model import TimeSeriesDataPreparator, LSTMModel
import os


class Backtester:
    """
    バックテストと評価を実行
    """
    def __init__(self, data: pd.DataFrame, feature_columns: list, lookback: int = 60):
        """
        Args:
            data: 処理済みデータ
            feature_columns: 特徴量のカラム名リスト
            lookback: 過去何日分のデータを使用するか
        """
        self.data = data.copy()
        self.feature_columns = feature_columns
        self.lookback = lookback
        self.data_preparator = TimeSeriesDataPreparator(lookback=lookback)
        
        # モデル
        self.regression_model = None
        self.classification_model = None
        self.peak_bottom_model = None
        
        # 予測結果
        self.predictions = {}
    
    def load_models(self, models_dir: str = "/Volumes/FUKUI-SSD01/fx_trade/models"):
        """
        学習済みモデルを読み込む
        """
        print("Loading models...")
        
        # 回帰モデル
        regression_path = os.path.join(models_dir, "price_regression_best.h5")
        if os.path.exists(regression_path):
            self.regression_model = LSTMModel((self.lookback, len(self.feature_columns)), "price_regression")
            self.regression_model.load_model(regression_path)
        
        # 分類モデル（方向予測）
        classification_path = os.path.join(models_dir, "direction_classification_best.h5")
        if os.path.exists(classification_path):
            self.classification_model = LSTMModel((self.lookback, len(self.feature_columns)), "direction_classification")
            self.classification_model.load_model(classification_path)
        
        # 天底検出モデル
        peak_bottom_path = os.path.join(models_dir, "peak_bottom_detection_best.h5")
        if os.path.exists(peak_bottom_path):
            self.peak_bottom_model = LSTMModel((self.lookback, len(self.feature_columns)), "peak_bottom_detection")
            self.peak_bottom_model.load_model(peak_bottom_path)
        
        print("Models loaded successfully!")
    
    def evaluate_regression_model(self, test_size: float = 0.2):
        """
        回帰モデルを評価
        """
        print("\n" + "=" * 80)
        print("Evaluating Regression Model (Price Prediction)")
        print("=" * 80)
        
        if self.regression_model is None:
            print("Regression model not loaded.")
            return
        
        # データを準備
        X_train, X_test, y_train, y_test = self.data_preparator.prepare_data(
            self.data, self.feature_columns, 'target_return', test_size
        )
        
        # 予測
        y_pred_train = self.regression_model.predict(X_train).flatten()
        y_pred_test = self.regression_model.predict(X_test).flatten()
        
        # 評価指標を計算
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)
        
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\nTrain Metrics:")
        print(f"  MAE: {train_mae:.6f}")
        print(f"  RMSE: {train_rmse:.6f}")
        print(f"  R2 Score: {train_r2:.6f}")
        
        print(f"\nTest Metrics:")
        print(f"  MAE: {test_mae:.6f}")
        print(f"  RMSE: {test_rmse:.6f}")
        print(f"  R2 Score: {test_r2:.6f}")
        
        # 予測結果を保存
        self.predictions['regression'] = {
            'y_test': y_test,
            'y_pred': y_pred_test,
            'metrics': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
        }
        
        # 可視化
        self._plot_regression_results(y_test, y_pred_test)
        
        return test_mae, test_rmse, test_r2
    
    def evaluate_classification_model(self, test_size: float = 0.2):
        """
        分類モデル（方向予測）を評価
        """
        print("\n" + "=" * 80)
        print("Evaluating Classification Model (Direction Prediction)")
        print("=" * 80)
        
        if self.classification_model is None:
            print("Classification model not loaded.")
            return
        
        # ターゲットを変換
        target_data = self.data['target_class'].copy() + 1
        self.data['target_class_mapped'] = target_data
        
        # データを準備
        X_train, X_test, y_train, y_test = self.data_preparator.prepare_data(
            self.data, self.feature_columns, 'target_class_mapped', test_size
        )
        
        # 予測
        y_pred_proba_test = self.classification_model.predict(X_test)
        y_pred_test = np.argmax(y_pred_proba_test, axis=1)
        
        # 評価
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['Down', 'Neutral', 'Up']))
        
        # 混同行列
        cm = confusion_matrix(y_test, y_pred_test)
        self._plot_confusion_matrix(cm, ['Down', 'Neutral', 'Up'], 'direction_classification')
        
        # 予測結果を保存
        self.predictions['classification'] = {
            'y_test': y_test,
            'y_pred': y_pred_test,
            'y_pred_proba': y_pred_proba_test
        }
        
        # 一時的な列を削除
        self.data.drop('target_class_mapped', axis=1, inplace=True)
        
        accuracy = np.mean(y_test == y_pred_test)
        return accuracy
    
    def evaluate_peak_bottom_model(self, test_size: float = 0.2):
        """
        天底検出モデルを評価
        """
        print("\n" + "=" * 80)
        print("Evaluating Peak/Bottom Detection Model")
        print("=" * 80)
        
        if self.peak_bottom_model is None:
            print("Peak/Bottom model not loaded.")
            return
        
        # データを準備
        X_train, X_test, y_train, y_test = self.data_preparator.prepare_data(
            self.data, self.feature_columns, 'target_peak_bottom', test_size
        )
        
        # 予測
        y_pred_proba_test = self.peak_bottom_model.predict(X_test)
        y_pred_test = np.argmax(y_pred_proba_test, axis=1)
        
        # 評価
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['Normal', 'Peak', 'Bottom']))
        
        # 混同行列
        cm = confusion_matrix(y_test, y_pred_test)
        self._plot_confusion_matrix(cm, ['Normal', 'Peak', 'Bottom'], 'peak_bottom_detection')
        
        # 予測結果を保存
        self.predictions['peak_bottom'] = {
            'y_test': y_test,
            'y_pred': y_pred_test,
            'y_pred_proba': y_pred_proba_test
        }
        
        accuracy = np.mean(y_test == y_pred_test)
        return accuracy
    
    def run_backtest(self, initial_capital: float = 1000000, test_size: float = 0.2):
        """
        トレーディング戦略のバックテストを実行
        
        Args:
            initial_capital: 初期資本
            test_size: テストデータの割合
        """
        print("\n" + "=" * 80)
        print("Running Backtest")
        print("=" * 80)
        
        if 'regression' not in self.predictions or 'classification' not in self.predictions:
            print("Please run model evaluation first.")
            return
        
        # データをクリーニング（評価時と同じ方法）
        # すべての特徴量とターゲット列を含むデータからNaNを除去
        required_cols = self.feature_columns + ['target_return']
        df_clean = self.data[required_cols].dropna()
        
        # 時系列データなので単純分割（最新データをテストに）
        # prepare_dataと同じロジックでインデックスを計算
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
        
        # closeカラムを元のデータから取得（念のため）
        if 'close' not in test_data.columns:
            test_data = test_data.join(self.data[['close']], how='left')
        
        # トレーディング戦略
        capital = initial_capital
        position = 0  # 0: なし, 1: 買いポジション
        shares = 0
        trades = []
        portfolio_values = [initial_capital]
        
        for i in range(len(test_data)):
            row = test_data.iloc[i]
            current_price = row['close']
            
            # 買いシグナル: 予測が上昇（2）で、かつ予測リターンがプラス
            if position == 0 and row['predicted_direction'] == 2 and row['predicted_return'] > 0.01:
                # 買い
                shares = capital / current_price
                position = 1
                trades.append({
                    'date': row.name,
                    'action': 'buy',
                    'price': current_price,
                    'shares': shares,
                    'capital': capital
                })
                print(f"BUY  {row.name.date()}: Price={current_price:.2f}, Shares={shares:.2f}")
            
            # 売りシグナル: 予測が下降（0）、または予測リターンがマイナス
            elif position == 1 and (row['predicted_direction'] == 0 or row['predicted_return'] < -0.005):
                # 売り
                capital = shares * current_price
                trades.append({
                    'date': row.name,
                    'action': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'capital': capital
                })
                print(f"SELL {row.name.date()}: Price={current_price:.2f}, Capital={capital:.2f}")
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
                'capital': capital
            })
        
        # パフォーマンス計算
        final_capital = portfolio_values[-1]
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        # Buy & Hold との比較
        buy_hold_return = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
        
        print(f"\n" + "=" * 80)
        print("Backtest Results")
        print("=" * 80)
        print(f"Initial Capital: ¥{initial_capital:,.0f}")
        print(f"Final Capital: ¥{final_capital:,.0f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"Outperformance: {total_return - buy_hold_return:.2f}%")
        print(f"Number of Trades: {len(trades)}")
        
        # 可視化
        self._plot_backtest_results(test_data, portfolio_values, trades, initial_capital)
        
        return {
            'final_capital': final_capital,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'trades': trades,
            'portfolio_values': portfolio_values
        }
    
    def _plot_regression_results(self, y_true, y_pred):
        """
        回帰モデルの結果をプロット
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 散布図
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Return')
        axes[0].set_ylabel('Predicted Return')
        axes[0].set_title('Actual vs Predicted Returns')
        axes[0].grid(True, alpha=0.3)
        
        # 時系列プロット
        axes[1].plot(y_true, label='Actual', alpha=0.7)
        axes[1].plot(y_pred, label='Predicted', alpha=0.7)
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('Return')
        axes[1].set_title('Time Series of Returns')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Volumes/FUKUI-SSD01/fx_trade/data/regression_evaluation.png', dpi=150)
        print("\nRegression evaluation plot saved to data/regression_evaluation.png")
        plt.close()
    
    def _plot_confusion_matrix(self, cm, labels, model_name):
        """
        混同行列をプロット
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(f'/Volumes/FUKUI-SSD01/fx_trade/data/{model_name}_confusion_matrix.png', dpi=150)
        print(f"\nConfusion matrix saved to data/{model_name}_confusion_matrix.png")
        plt.close()
    
    def _plot_backtest_results(self, test_data, portfolio_values, trades, initial_capital):
        """
        バックテスト結果をプロット
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # 株価と売買シグナル
        axes[0].plot(test_data.index, test_data['close'], label='Stock Price', color='black', linewidth=1.5)
        
        # 買いシグナルと売りシグナルをマーク
        for trade in trades:
            if trade['action'] == 'buy':
                axes[0].scatter(trade['date'], trade['price'], color='green', marker='^', s=150, zorder=5, label='Buy' if 'Buy' not in [t.get_label() for t in axes[0].get_children()] else '')
            else:
                axes[0].scatter(trade['date'], trade['price'], color='red', marker='v', s=150, zorder=5, label='Sell' if 'Sell' not in [t.get_label() for t in axes[0].get_children()] else '')
        
        axes[0].set_ylabel('Price')
        axes[0].set_title('Stock Price with Trading Signals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # ポートフォリオ価値の推移
        axes[1].plot(test_data.index, portfolio_values[1:], label='Portfolio Value', color='blue', linewidth=2)
        axes[1].axhline(y=initial_capital, color='gray', linestyle='--', label='Initial Capital')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Portfolio Value (¥)')
        axes[1].set_title('Portfolio Value Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Volumes/FUKUI-SSD01/fx_trade/data/backtest_results.png', dpi=150)
        print("\nBacktest results plot saved to data/backtest_results.png")
        plt.close()


if __name__ == "__main__":
    # テスト用
    pass

