#!/usr/bin/env python3
"""
Focal Loss + 時系列CV統合トレーナー
根本的な改善を実装
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import keras

# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.feature_engineering import FeatureEngineering
from src.model import LSTMModel, TimeSeriesDataPreparator
from src.focal_loss import categorical_focal_loss, binary_focal_loss
from src.time_series_cv import TimeSeriesCrossValidator, MarketEnvironmentSplitter, analyze_data_splits

class FocalLossTrainer:
    """
    Focal Loss + 時系列CV統合トレーナー
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
        
        # データ
        self.data = None
        self.feature_columns = None
        
        # モデル
        self.regression_model = None
        self.classification_model = None
        self.peak_bottom_model = None
        
        # 時系列CV（より適切な設定）
        self.tscv = TimeSeriesCrossValidator(n_splits=3, test_size=0.3)
        self.mes = MarketEnvironmentSplitter()
        
        # 結果保存
        self.results = {}
    
    def load_and_prepare_data(self, period: str = "10y"):
        """
        データの読み込みと前処理
        """
        print("=" * 80)
        print("Step 1: Loading and preparing data...")
        print("=" * 80)
        
        # データ取得
        fetcher = DataFetcher(self.ticker, period=period)
        self.data = fetcher.fetch_data()
        
        # 特徴量エンジニアリング
        engineer = FeatureEngineering(self.data)
        self.data = engineer.add_technical_indicators()
        self.data = engineer.detect_peaks_and_bottoms()
        self.data = engineer.create_target_labels()
        self.data = engineer.create_time_series_features()
        self.data = engineer.handle_missing_values()
        
        print(f"Data shape: {self.data.shape}")
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
    
    def calculate_class_weights(self, y_train: np.ndarray, num_classes: int) -> dict:
        """
        クラスウェイトを計算（Focal Loss用）
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # クラスラベルを0から始まるように変換
        y_train_shifted = y_train - np.min(y_train)
        
        # クラス分布を計算
        class_counts = np.bincount(y_train_shifted.astype(int))
        total_samples = len(y_train)
        
        # クラスウェイトを計算（より極端に）
        class_weights = {}
        for i in range(num_classes):
            if i < len(class_counts) and class_counts[i] > 0:
                # 不均衡度に基づいて重みを計算
                imbalance_ratio = total_samples / class_counts[i]
                # より極端な重み付け（最大50倍）
                weight = min(imbalance_ratio * 2, 50.0)
                class_weights[i] = weight
            else:
                class_weights[i] = 1.0
        
        print(f"Class weights: {class_weights}")
        return class_weights
    
    def train_with_focal_loss(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_val: np.ndarray, y_val: np.ndarray,
                             model_type: str, epochs: int = 100, batch_size: int = 32,
                             y_train_raw: np.ndarray = None):
        """
        Focal Lossでモデルを学習
        
        Args:
            model_type: 'regression', 'classification', 'peak_bottom'
        """
        print(f"\nTraining {model_type} model with Focal Loss...")
        
        # モデルを構築
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = LSTMModel(input_shape, model_name=f"{model_type}_focal")
        
        if model_type == 'regression':
            # 回帰モデル
            model.build_regression_model(lstm_units=[128, 64], dropout_rate=0.3)
            
            # 回帰用のFocal Loss（MSEベース）
            model.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
        elif model_type == 'classification':
            # 分類モデル（方向予測）
            model.build_classification_model(num_classes=3, lstm_units=[128, 64], dropout_rate=0.3)
            
            # クラスウェイトを計算（元のラベルを使用）
            if y_train_raw is not None:
                class_weights = self.calculate_class_weights(y_train_raw, 3)
                alpha = [class_weights.get(i, 1.0) for i in range(3)]
            else:
                class_weights = None
                alpha = None
            
            # 通常のCross Entropy Lossを使用
            model.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
        elif model_type == 'peak_bottom':
            # 天底検出モデル
            model.build_classification_model(num_classes=3, lstm_units=[128, 64], dropout_rate=0.3)
            
            # クラスウェイトを計算（元のラベルを使用）
            if y_train_raw is not None:
                class_weights = self.calculate_class_weights(y_train_raw, 3)
                alpha = [class_weights.get(i, 1.0) for i in range(3)]
            else:
                class_weights = None
                alpha = None
            
            # 通常のCross Entropy Lossを使用
            model.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # 学習
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            keras.callbacks.ModelCheckpoint(
                f'models/{model_type}_focal_best.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        history = model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 学習履歴をプロット
        self.plot_training_history(history, model_type)
        
        # モデルを保存
        model.save_model()
        
        return model, history
    
    def plot_training_history(self, history, model_type: str):
        """
        学習履歴をプロット
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title(f'{model_type.title()} Model - Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Metrics
        if 'accuracy' in history.history:
            axes[1].plot(history.history['accuracy'], label='Training Accuracy')
            axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[1].set_title(f'{model_type.title()} Model - Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        elif 'mae' in history.history:
            axes[1].plot(history.history['mae'], label='Training MAE')
            axes[1].plot(history.history['val_mae'], label='Validation MAE')
            axes[1].set_title(f'{model_type.title()} Model - MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'data/{model_type}_focal_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_with_time_series_cv(self, epochs: int = 100, batch_size: int = 32):
        """
        時系列クロスバリデーションで学習
        """
        print("=" * 80)
        print("Step 2: Training with Time Series Cross Validation...")
        print("=" * 80)
        
        # データをクリーニング
        required_cols = self.feature_columns + ['target_return', 'target_class', 'target_peak_bottom']
        df_clean = self.data[required_cols].dropna()
        
        print(f"Clean data shape: {df_clean.shape}")
        
        # 時系列CVで分割
        splits = list(self.tscv.split(df_clean))
        print(f"Generated {len(splits)} time series splits")
        
        # 分割を分析
        analyze_data_splits(df_clean, splits, [f"TS-CV {i+1}" for i in range(len(splits))])
        
        # 各foldで学習・評価
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}/{len(splits)}")
            print(f"{'='*60}")
            
            # データを分割
            train_data = df_clean.iloc[train_idx]
            val_data = df_clean.iloc[val_idx]
            
            print(f"Train: {len(train_data)} samples")
            print(f"Validation: {len(val_data)} samples")
            
            # データを準備
            preparator = TimeSeriesDataPreparator(lookback=self.lookback)
            
            # 回帰データ
            X_train_reg, _, y_train_reg, _ = preparator.prepare_data(
                train_data, self.feature_columns, 'target_return', test_size=0.0
            )
            X_val_reg, _, y_val_reg, _ = preparator.prepare_data(
                val_data, self.feature_columns, 'target_return', test_size=0.0
            )
            
            # 分類データ（方向）- one-hotエンコード
            X_train_class, _, y_train_class_raw, _ = preparator.prepare_data(
                train_data, self.feature_columns, 'target_class', test_size=0.0
            )
            X_val_class, _, y_val_class_raw, _ = preparator.prepare_data(
                val_data, self.feature_columns, 'target_class', test_size=0.0
            )
            
            # one-hotエンコードに変換
            y_train_class = tf.keras.utils.to_categorical(y_train_class_raw, num_classes=3)
            y_val_class = tf.keras.utils.to_categorical(y_val_class_raw, num_classes=3)
            
            # 天底データ - one-hotエンコード
            X_train_peak, _, y_train_peak_raw, _ = preparator.prepare_data(
                train_data, self.feature_columns, 'target_peak_bottom', test_size=0.0
            )
            X_val_peak, _, y_val_peak_raw, _ = preparator.prepare_data(
                val_data, self.feature_columns, 'target_peak_bottom', test_size=0.0
            )
            
            # one-hotエンコードに変換
            y_train_peak = tf.keras.utils.to_categorical(y_train_peak_raw, num_classes=3)
            y_val_peak = tf.keras.utils.to_categorical(y_val_peak_raw, num_classes=3)
            
            print(f"Regression data: X_train{X_train_reg.shape}, y_train{y_train_reg.shape}")
            print(f"Classification data: X_train{X_train_class.shape}, y_train{y_train_class.shape}")
            print(f"Peak/Bottom data: X_train{X_train_peak.shape}, y_train{y_train_peak.shape}")
            
            # モデルを学習（最初のfoldのみ）
            if fold_idx == 0:
                # 回帰モデル
                self.regression_model, reg_history = self.train_with_focal_loss(
                    X_train_reg, y_train_reg, X_val_reg, y_val_reg,
                    'regression', epochs=epochs, batch_size=batch_size
                )
                
                # 分類モデル（方向）
                self.classification_model, class_history = self.train_with_focal_loss(
                    X_train_class, y_train_class, X_val_class, y_val_class,
                    'classification', epochs=epochs, batch_size=batch_size,
                    y_train_raw=y_train_class_raw
                )
                
                # 天底モデル
                self.peak_bottom_model, peak_history = self.train_with_focal_loss(
                    X_train_peak, y_train_peak, X_val_peak, y_val_peak,
                    'peak_bottom', epochs=epochs, batch_size=batch_size,
                    y_train_raw=y_train_peak_raw
                )
            
            # 評価（全fold）
            fold_result = self.evaluate_fold(
                fold_idx, X_val_reg, y_val_reg, X_val_class, y_val_class, 
                X_val_peak, y_val_peak
            )
            fold_results.append(fold_result)
        
        # 全foldの結果をまとめる
        self.summarize_cv_results(fold_results)
        
        return fold_results
    
    def evaluate_fold(self, fold_idx: int, X_val_reg: np.ndarray, y_val_reg: np.ndarray,
                     X_val_class: np.ndarray, y_val_class: np.ndarray,
                     X_val_peak: np.ndarray, y_val_peak: np.ndarray) -> dict:
        """
        各foldの評価
        """
        if fold_idx == 0:  # 最初のfoldのみ評価（モデルが学習済みの場合）
            # 予測
            y_pred_reg = self.regression_model.model.predict(X_val_reg, verbose=0)
            y_pred_class = self.classification_model.model.predict(X_val_class, verbose=0)
            y_pred_peak = self.peak_bottom_model.model.predict(X_val_peak, verbose=0)
            
            # 分類予測をargmaxに変換
            y_pred_class = np.argmax(y_pred_class, axis=1)
            y_pred_peak = np.argmax(y_pred_peak, axis=1)
            
            # 真の値もone-hotから整数ラベルに変換
            y_true_class = np.argmax(y_val_class, axis=1)
            y_true_peak = np.argmax(y_val_peak, axis=1)
            
            # 評価指標を計算
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.metrics import accuracy_score, classification_report
            
            reg_mae = mean_absolute_error(y_val_reg, y_pred_reg)
            reg_mse = mean_squared_error(y_val_reg, y_pred_reg)
            reg_r2 = r2_score(y_val_reg, y_pred_reg)
            
            class_acc = accuracy_score(y_true_class, y_pred_class)
            peak_acc = accuracy_score(y_true_peak, y_pred_peak)
            
            result = {
                'fold': fold_idx,
                'regression': {'mae': reg_mae, 'mse': reg_mse, 'r2': reg_r2},
                'classification': {'accuracy': class_acc},
                'peak_bottom': {'accuracy': peak_acc},
                'predictions': {
                    'regression': y_pred_reg.flatten(),
                    'classification': y_pred_class,
                    'peak_bottom': y_pred_peak
                }
            }
            
            print(f"Fold {fold_idx + 1} Results:")
            print(f"  Regression - MAE: {reg_mae:.4f}, MSE: {reg_mse:.4f}, R2: {reg_r2:.4f}")
            print(f"  Classification - Accuracy: {class_acc:.4f}")
            print(f"  Peak/Bottom - Accuracy: {peak_acc:.4f}")
            
            return result
        else:
            return {'fold': fold_idx, 'skipped': True}
    
    def summarize_cv_results(self, fold_results: list):
        """
        クロスバリデーション結果をまとめる
        """
        print("\n" + "=" * 80)
        print("Cross Validation Results Summary")
        print("=" * 80)
        
        # 有効な結果のみを抽出
        valid_results = [r for r in fold_results if 'skipped' not in r]
        
        if valid_results:
            # 回帰結果
            reg_maes = [r['regression']['mae'] for r in valid_results]
            reg_mses = [r['regression']['mse'] for r in valid_results]
            reg_r2s = [r['regression']['r2'] for r in valid_results]
            
            print(f"\nRegression Results:")
            print(f"  MAE: {np.mean(reg_maes):.4f} ± {np.std(reg_maes):.4f}")
            print(f"  MSE: {np.mean(reg_mses):.4f} ± {np.std(reg_mses):.4f}")
            print(f"  R2: {np.mean(reg_r2s):.4f} ± {np.std(reg_r2s):.4f}")
            
            # 分類結果
            class_accs = [r['classification']['accuracy'] for r in valid_results]
            peak_accs = [r['peak_bottom']['accuracy'] for r in valid_results]
            
            print(f"\nClassification Results:")
            print(f"  Direction Accuracy: {np.mean(class_accs):.4f} ± {np.std(class_accs):.4f}")
            print(f"  Peak/Bottom Accuracy: {np.mean(peak_accs):.4f} ± {np.std(peak_accs):.4f}")
        
        self.results['cv_results'] = fold_results
    
    def train_all_models_focal(self, epochs: int = 100, batch_size: int = 32):
        """
        Focal Loss + 時系列CVで全モデルを学習
        """
        # データの準備
        self.load_and_prepare_data(period="10y")
        self.select_features()
        
        # 時系列CVで学習
        cv_results = self.train_with_time_series_cv(epochs=epochs, batch_size=batch_size)
        
        print("\n" + "=" * 80)
        print("Focal Loss + Time Series CV Training Completed!")
        print("=" * 80)
        
        return cv_results

def main():
    """
    メイン実行関数
    """
    print("=" * 80)
    print("Focal Loss + 時系列CV 統合トレーナー")
    print("=" * 80)
    
    # トレーナーを初期化
    trainer = FocalLossTrainer("7203.T", lookback=60, test_size=0.2)
    
    # 学習実行
    results = trainer.train_all_models_focal(epochs=100, batch_size=32)
    
    print("\n🎉 学習完了！")
    print("📊 結果:")
    print("  - Focal Loss適用")
    print("  - 時系列クロスバリデーション")
    print("  - 極端なクラスウェイト")
    print("  - モデル: models/*_focal_best.h5")
    print("  - 履歴: data/*_focal_history.png")

if __name__ == "__main__":
    main()
