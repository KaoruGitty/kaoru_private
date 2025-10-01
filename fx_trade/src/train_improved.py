"""
改善版モデル学習スクリプト
- データ期間を10年に拡大
- クラスウェイトの適用
- より厳しい閾値設定
"""
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from data_fetcher import DataFetcher
from feature_engineering import FeatureEngineering
from model import TimeSeriesDataPreparator, LSTMModel
import os


class ImprovedModelTrainer:
    """
    改善版モデルトレーナー
    """
    def __init__(self, ticker: str, lookback: int = 60, test_size: float = 0.2):
        self.ticker = ticker
        self.lookback = lookback
        self.test_size = test_size
        self.data = None
        self.feature_columns = None
        self.data_preparator = TimeSeriesDataPreparator(lookback=lookback)
        
        # クラスウェイト
        self.class_weights_direction = None
        self.class_weights_peak_bottom = None
        
        # モデル
        self.regression_model = None
        self.classification_model = None
        self.peak_bottom_model = None
    
    def load_and_prepare_data(self, data_path: str = None, period: str = "10y"):
        """
        データの読み込みと前処理（10年分に拡大）
        """
        print("=" * 80)
        print("Step 1: Loading and preparing data (10 years)...")
        print("=" * 80)
        
        if data_path and os.path.exists(data_path):
            self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"Data loaded from {data_path}")
        else:
            # データを取得（10年分）
            fetcher = DataFetcher(self.ticker, period=period)
            data = fetcher.fetch_data()
            
            # 特徴量エンジニアリング
            fe = FeatureEngineering(data)
            
            # テクニカル指標を追加
            fe.add_technical_indicators()
            
            # より厳しい閾値で天底を検出（ノイズを減らす）
            fe.detect_peaks_and_bottoms(window=7)  # 5 → 7日に変更
            
            # 買いシグナルを生成（より厳しい閾値）
            fe.create_buy_signals(threshold_return=0.03)  # 0.02 → 0.03に変更
            
            # 正解ラベルを作成（より厳しい閾値）
            fe.create_target_labels(horizon=5, price_change_threshold=0.04)  # 0.03 → 0.04に変更
            
            # 時系列特徴量を作成
            fe.create_time_series_features()
            
            # 欠損値を補完
            fe.handle_missing_values()
            
            # 可視化
            fe.visualize_signals(start_idx=0, end_idx=min(500, len(fe.data)))
            
            self.data = fe.data
            
            # データを保存
            save_path = data_path if data_path else '/Volumes/FUKUI-SSD01/fx_trade/data/processed_data_improved.csv'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.data.to_csv(save_path)
            print(f"Processed data saved to {save_path}")
        
        print(f"\nData shape: {self.data.shape}")
        print(f"Data period: {self.data.index[0]} to {self.data.index[-1]}")
        print(f"Years: {(self.data.index[-1] - self.data.index[0]).days / 365:.1f}")
        
        # クラスバランスを表示
        self._analyze_class_balance()
        
        return self.data
    
    def _analyze_class_balance(self):
        """
        クラスバランスを分析して表示
        """
        print(f"\n{'='*80}")
        print("Class Balance Analysis")
        print('='*80)
        
        if 'target_class' in self.data.columns:
            class_dist = self.data['target_class'].value_counts().sort_index()
            total = len(self.data.dropna(subset=['target_class']))
            print(f"\n【Price Direction (target_class)】")
            print(f"  Down (-1):  {class_dist.get(-1, 0):4d} ({class_dist.get(-1, 0)/total*100:5.1f}%)")
            print(f"  Neutral (0): {class_dist.get(0, 0):4d} ({class_dist.get(0, 0)/total*100:5.1f}%)")
            print(f"  Up (1):     {class_dist.get(1, 0):4d} ({class_dist.get(1, 0)/total*100:5.1f}%)")
            print(f"  Imbalance ratio: {class_dist.max() / class_dist.min():.1f}x")
            
            # クラスウェイトを計算
            classes = np.array([-1, 0, 1])
            weights = compute_class_weight('balanced', classes=classes, 
                                          y=self.data['target_class'].dropna())
            self.class_weights_direction = dict(zip([0, 1, 2], weights))  # -1,0,1 → 0,1,2にマッピング
            print(f"  Computed class weights: {self.class_weights_direction}")
        
        if 'target_peak_bottom' in self.data.columns:
            pb_dist = self.data['target_peak_bottom'].value_counts().sort_index()
            total = len(self.data.dropna(subset=['target_peak_bottom']))
            print(f"\n【Peak/Bottom Detection (target_peak_bottom)】")
            print(f"  Normal (0): {pb_dist.get(0, 0):4d} ({pb_dist.get(0, 0)/total*100:5.1f}%)")
            print(f"  Peak (1):   {pb_dist.get(1, 0):4d} ({pb_dist.get(1, 0)/total*100:5.1f}%)")
            print(f"  Bottom (2): {pb_dist.get(2, 0):4d} ({pb_dist.get(2, 0)/total*100:5.1f}%)")
            print(f"  Imbalance ratio: {pb_dist.max() / pb_dist.min():.1f}x")
            
            # クラスウェイトを計算
            classes = np.array([0, 1, 2])
            weights = compute_class_weight('balanced', classes=classes, 
                                          y=self.data['target_peak_bottom'].dropna())
            self.class_weights_peak_bottom = dict(zip(classes, weights))
            print(f"  Computed class weights: {self.class_weights_peak_bottom}")
        
        print('='*80)
    
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
    
    def train_classification_model(self, epochs: int = 100, batch_size: int = 32):
        """
        価格変動方向予測（分類）モデルを学習（クラスウェイト適用）
        """
        print("\n" + "=" * 80)
        print("Step 2: Training classification model with class weights...")
        print("=" * 80)
        
        # ターゲットを変換
        target_data = self.data['target_class'].copy() + 1
        self.data['target_class_mapped'] = target_data
        
        # データを準備
        X_train, X_test, y_train, y_test = self.data_preparator.prepare_data(
            self.data, self.feature_columns, 'target_class_mapped', self.test_size
        )
        
        # モデルを構築
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.classification_model = LSTMModel(input_shape, model_name="direction_classification_improved")
        self.classification_model.build_classification_model(num_classes=3, lstm_units=[128, 64], dropout_rate=0.3)
        
        # クラスウェイトを適用して学習
        print(f"\nApplying class weights: {self.class_weights_direction}")
        self.classification_model.train(
            X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size,
            class_weight=self.class_weights_direction  # クラスウェイトを適用
        )
        
        # 学習履歴をプロット
        self.classification_model.plot_training_history()
        
        # モデルを保存
        self.classification_model.save_model()
        
        # 一時的な列を削除
        self.data.drop('target_class_mapped', axis=1, inplace=True)
        
        print("\nImproved classification model training completed!")
        
        return self.classification_model
    
    def train_peak_bottom_model(self, epochs: int = 100, batch_size: int = 32):
        """
        天底検出（分類）モデルを学習（クラスウェイト適用）
        """
        print("\n" + "=" * 80)
        print("Step 3: Training peak/bottom model with class weights...")
        print("=" * 80)
        
        # データを準備
        X_train, X_test, y_train, y_test = self.data_preparator.prepare_data(
            self.data, self.feature_columns, 'target_peak_bottom', self.test_size
        )
        
        # モデルを構築
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.peak_bottom_model = LSTMModel(input_shape, model_name="peak_bottom_detection_improved")
        self.peak_bottom_model.build_classification_model(num_classes=3, lstm_units=[128, 64], dropout_rate=0.3)
        
        # クラスウェイトを適用して学習
        print(f"\nApplying class weights: {self.class_weights_peak_bottom}")
        self.peak_bottom_model.train(
            X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size,
            class_weight=self.class_weights_peak_bottom  # クラスウェイトを適用
        )
        
        # 学習履歴をプロット
        self.peak_bottom_model.plot_training_history()
        
        # モデルを保存
        self.peak_bottom_model.save_model()
        
        print("\nImproved peak/bottom detection model training completed!")
        
        return self.peak_bottom_model
    
    def train_all_models_improved(self, epochs: int = 100, batch_size: int = 32):
        """
        改善版：全モデルを学習
        """
        # データの準備
        self.load_and_prepare_data(period="10y")
        
        # 特徴量を選択
        self.select_features()
        
        # 各モデルを学習（クラスウェイト適用）
        self.train_classification_model(epochs=epochs, batch_size=batch_size)
        self.train_peak_bottom_model(epochs=epochs, batch_size=batch_size)
        
        print("\n" + "=" * 80)
        print("All improved models training completed!")
        print("=" * 80)


if __name__ == "__main__":
    # 改善版でテスト
    trainer = ImprovedModelTrainer("7203.T", lookback=60, test_size=0.2)
    trainer.train_all_models_improved(epochs=100, batch_size=32)

