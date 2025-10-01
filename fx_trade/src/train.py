"""
モデルの学習を実行するスクリプト
"""
import pandas as pd
import numpy as np
from data_fetcher import DataFetcher
from feature_engineering import FeatureEngineering
from model import TimeSeriesDataPreparator, LSTMModel
import os


class ModelTrainer:
    """
    モデルの学習を統合的に管理
    """
    def __init__(self, ticker: str, lookback: int = 60, test_size: float = 0.2):
        """
        Args:
            ticker: 銘柄コード
            lookback: 過去何日分のデータを使用するか
            test_size: テストデータの割合
        """
        self.ticker = ticker
        self.lookback = lookback
        self.test_size = test_size
        self.data = None
        self.feature_columns = None
        self.data_preparator = TimeSeriesDataPreparator(lookback=lookback)
        
        # モデル
        self.regression_model = None
        self.classification_model = None
        self.peak_bottom_model = None
    
    def load_and_prepare_data(self, data_path: str = None):
        """
        データの読み込みと前処理
        """
        print("=" * 80)
        print("Step 1: Loading and preparing data...")
        print("=" * 80)
        
        if data_path and os.path.exists(data_path):
            # 保存済みデータを読み込む
            self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"Data loaded from {data_path}")
        else:
            # データを取得
            fetcher = DataFetcher(self.ticker, period="5y")
            data = fetcher.fetch_data()
            
            # 特徴量エンジニアリング
            fe = FeatureEngineering(data)
            
            # テクニカル指標を追加
            fe.add_technical_indicators()
            
            # 天底を検出
            fe.detect_peaks_and_bottoms(window=5)
            
            # 買いシグナルを生成
            fe.create_buy_signals(threshold_return=0.02)
            
            # 正解ラベルを作成
            fe.create_target_labels(horizon=5, price_change_threshold=0.03)
            
            # 時系列特徴量を作成
            fe.create_time_series_features()
            
            # 欠損値を補完
            fe.handle_missing_values()
            
            # 可視化
            fe.visualize_signals(start_idx=0, end_idx=min(500, len(fe.data)))
            
            self.data = fe.data
            
            # データを保存
            save_path = data_path if data_path else '/Volumes/FUKUI-SSD01/fx_trade/data/processed_data.csv'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.data.to_csv(save_path)
            print(f"Processed data saved to {save_path}")
        
        print(f"\nData shape: {self.data.shape}")
        print(f"Columns: {len(self.data.columns)}")
        
        return self.data
    
    def select_features(self):
        """
        学習に使用する特徴量を選択
        """
        # 基本的なOHLCV
        base_features = ['open', 'high', 'low', 'close', 'volume']
        
        # テクニカル指標
        technical_features = [col for col in self.data.columns if any(
            indicator in col for indicator in [
                'sma', 'ema', 'macd', 'rsi', 'stoch', 'bb', 'atr', 'obv',
                'return', 'volatility', 'volume_sma', 'volume_ratio'
            ]
        )]
        
        # 時系列特徴量
        time_features = ['day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter']
        
        # 全特徴量
        self.feature_columns = base_features + technical_features + time_features
        
        # 存在しない列を除外
        self.feature_columns = [col for col in self.feature_columns if col in self.data.columns]
        
        # ターゲット変数や不要な列を除外
        exclude_columns = ['next_day_return', 'target_return', 'target_class', 'target_class_mapped',
                          'target_peak_bottom', 'is_peak', 'is_bottom', 'buy_signal']
        self.feature_columns = [col for col in self.feature_columns if col not in exclude_columns]
        
        print(f"\nSelected {len(self.feature_columns)} features for training")
        
        return self.feature_columns
    
    def train_regression_model(self, epochs: int = 100, batch_size: int = 32):
        """
        価格変化率予測（回帰）モデルを学習
        """
        print("\n" + "=" * 80)
        print("Step 2: Training regression model (price prediction)...")
        print("=" * 80)
        
        # データを準備
        X_train, X_test, y_train, y_test = self.data_preparator.prepare_data(
            self.data, self.feature_columns, 'target_return', self.test_size
        )
        
        # モデルを構築
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.regression_model = LSTMModel(input_shape, model_name="price_regression")
        self.regression_model.build_regression_model(lstm_units=[128, 64], dropout_rate=0.3)
        
        # 学習
        self.regression_model.train(
            X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size
        )
        
        # 学習履歴をプロット
        self.regression_model.plot_training_history()
        
        # モデルを保存
        self.regression_model.save_model()
        
        print("\nRegression model training completed!")
        
        return self.regression_model
    
    def train_classification_model(self, epochs: int = 100, batch_size: int = 32):
        """
        価格変動方向予測（分類）モデルを学習
        """
        print("\n" + "=" * 80)
        print("Step 3: Training classification model (direction prediction)...")
        print("=" * 80)
        
        # ターゲットを-1,0,1から0,1,2に変換（sparseのため）
        target_data = self.data['target_class'].copy()
        target_data = target_data + 1  # -1,0,1 -> 0,1,2
        
        # 一時的にデータフレームに追加
        self.data['target_class_mapped'] = target_data
        
        # データを準備
        X_train, X_test, y_train, y_test = self.data_preparator.prepare_data(
            self.data, self.feature_columns, 'target_class_mapped', self.test_size
        )
        
        # モデルを構築
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.classification_model = LSTMModel(input_shape, model_name="direction_classification")
        self.classification_model.build_classification_model(num_classes=3, lstm_units=[128, 64], dropout_rate=0.3)
        
        # 学習
        self.classification_model.train(
            X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size
        )
        
        # 学習履歴をプロット
        self.classification_model.plot_training_history()
        
        # モデルを保存
        self.classification_model.save_model()
        
        # 一時的な列を削除
        self.data.drop('target_class_mapped', axis=1, inplace=True)
        
        print("\nClassification model training completed!")
        
        return self.classification_model
    
    def train_peak_bottom_model(self, epochs: int = 100, batch_size: int = 32):
        """
        天底検出（分類）モデルを学習
        """
        print("\n" + "=" * 80)
        print("Step 4: Training peak/bottom detection model...")
        print("=" * 80)
        
        # データを準備
        X_train, X_test, y_train, y_test = self.data_preparator.prepare_data(
            self.data, self.feature_columns, 'target_peak_bottom', self.test_size
        )
        
        # モデルを構築
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.peak_bottom_model = LSTMModel(input_shape, model_name="peak_bottom_detection")
        self.peak_bottom_model.build_classification_model(num_classes=3, lstm_units=[128, 64], dropout_rate=0.3)
        
        # 学習
        self.peak_bottom_model.train(
            X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size
        )
        
        # 学習履歴をプロット
        self.peak_bottom_model.plot_training_history()
        
        # モデルを保存
        self.peak_bottom_model.save_model()
        
        print("\nPeak/bottom detection model training completed!")
        
        return self.peak_bottom_model
    
    def train_all_models(self, epochs: int = 100, batch_size: int = 32):
        """
        全モデルを学習
        """
        # データの準備
        self.load_and_prepare_data()
        
        # 特徴量を選択
        self.select_features()
        
        # 各モデルを学習
        self.train_regression_model(epochs=epochs, batch_size=batch_size)
        self.train_classification_model(epochs=epochs, batch_size=batch_size)
        self.train_peak_bottom_model(epochs=epochs, batch_size=batch_size)
        
        print("\n" + "=" * 80)
        print("All models training completed!")
        print("=" * 80)


if __name__ == "__main__":
    # デフォルトの銘柄でテスト
    trainer = ModelTrainer("7203.T", lookback=60, test_size=0.2)
    trainer.train_all_models(epochs=50, batch_size=32)

