"""
LSTMベースの時系列予測モデル
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


class TimeSeriesDataPreparator:
    """
    時系列データをLSTM用に変換
    """
    def __init__(self, lookback: int = 60):
        """
        Args:
            lookback: 過去何日分のデータを使用するか
        """
        self.lookback = lookback
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray):
        """
        時系列データをシーケンスに変換
        
        Args:
            data: 特徴量データ (samples, features)
            target: ターゲットデータ (samples,)
        
        Returns:
            X: (samples, lookback, features)
            y: (samples,)
        """
        X, y = [], []
        
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df: pd.DataFrame, feature_columns: list, target_column: str, test_size: float = 0.2):
        """
        データを学習用と検証用に分割
        
        Args:
            df: データフレーム
            feature_columns: 特徴量のカラム名リスト
            target_column: ターゲットのカラム名
            test_size: テストデータの割合
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # NaNを含む行を削除
        df_clean = df[feature_columns + [target_column]].dropna()
        
        # データを抽出（明示的にfloat32に変換）
        X_data = df_clean[feature_columns].values.astype(np.float32)
        y_data = df_clean[target_column].values.astype(np.float32)
        
        # シーケンスを作成
        X, y = self.create_sequences(X_data, y_data)
        
        # 時系列データなので単純分割（最新データをテストに）
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx].astype(np.float32), X[split_idx:].astype(np.float32)
        y_train, y_test = y[:split_idx].astype(np.float32), y[split_idx:].astype(np.float32)
        
        print(f"Data prepared:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test


class LSTMModel:
    """
    LSTM モデルのベースクラス
    """
    def __init__(self, input_shape: tuple, model_name: str = "lstm_model"):
        """
        Args:
            input_shape: (lookback, features)
            model_name: モデル名
        """
        self.input_shape = input_shape
        self.model_name = model_name
        self.model = None
        self.history = None
    
    def build_regression_model(self, lstm_units: list = [128, 64], dropout_rate: float = 0.2):
        """
        回帰モデルを構築（価格変化率予測）
        
        Args:
            lstm_units: 各LSTM層のユニット数
            dropout_rate: ドロップアウト率
        """
        model = keras.Sequential(name=f"{self.model_name}_regression")
        
        # 最初のLSTM層
        model.add(layers.LSTM(lstm_units[0], return_sequences=True, input_shape=self.input_shape))
        model.add(layers.Dropout(dropout_rate))
        
        # 中間LSTM層
        for units in lstm_units[1:-1]:
            model.add(layers.LSTM(units, return_sequences=True))
            model.add(layers.Dropout(dropout_rate))
        
        # 最後のLSTM層
        model.add(layers.LSTM(lstm_units[-1], return_sequences=False))
        model.add(layers.Dropout(dropout_rate))
        
        # 全結合層
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        
        # 出力層（回帰）
        model.add(layers.Dense(1, activation='linear'))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.model = model
        print(f"Regression model built:")
        model.summary()
        
        return model
    
    def build_classification_model(self, num_classes: int, lstm_units: list = [128, 64], dropout_rate: float = 0.2):
        """
        分類モデルを構築（天底検出、上昇/下降予測）
        
        Args:
            num_classes: クラス数
            lstm_units: 各LSTM層のユニット数
            dropout_rate: ドロップアウト率
        """
        model = keras.Sequential(name=f"{self.model_name}_classification")
        
        # 最初のLSTM層
        model.add(layers.LSTM(lstm_units[0], return_sequences=True, input_shape=self.input_shape))
        model.add(layers.Dropout(dropout_rate))
        
        # 中間LSTM層
        for units in lstm_units[1:-1]:
            model.add(layers.LSTM(units, return_sequences=True))
            model.add(layers.Dropout(dropout_rate))
        
        # 最後のLSTM層
        model.add(layers.LSTM(lstm_units[-1], return_sequences=False))
        model.add(layers.Dropout(dropout_rate))
        
        # 全結合層
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        
        # 出力層（分類）
        activation = 'softmax' if num_classes > 2 else 'sigmoid'
        output_units = num_classes if num_classes > 2 else 1
        
        model.add(layers.Dense(output_units, activation=activation))
        
        loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        
        self.model = model
        print(f"Classification model built ({num_classes} classes):")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 32, class_weight=None):
        """
        モデルを学習
        
        Args:
            X_train, y_train: 学習データ
            X_val, y_val: 検証データ
            epochs: エポック数
            batch_size: バッチサイズ
            class_weight: クラスウェイト（不均衡データ対策）
        """
        if self.model is None:
            raise ValueError("Model not built. Please call build_regression_model or build_classification_model first.")
        
        # コールバック設定
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(
                f'/Volumes/FUKUI-SSD01/fx_trade/models/{self.model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 学習実行
        print(f"\nTraining {self.model_name}...")
        if class_weight:
            print(f"Using class weights: {class_weight}")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,  # クラスウェイトを追加
            verbose=1
        )
        
        print(f"\nTraining completed for {self.model_name}")
        
        return self.history
    
    def plot_training_history(self):
        """
        学習履歴をプロット
        """
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.model_name} - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metric
        metric_key = 'mae' if 'mae' in self.history.history else 'accuracy'
        axes[1].plot(self.history.history[metric_key], label=f'Train {metric_key}')
        axes[1].plot(self.history.history[f'val_{metric_key}'], label=f'Val {metric_key}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_key.upper())
        axes[1].set_title(f'{self.model_name} - {metric_key.upper()}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/Volumes/FUKUI-SSD01/fx_trade/data/{self.model_name}_history.png', dpi=150)
        print(f"Training history saved to data/{self.model_name}_history.png")
        plt.close()
    
    def save_model(self, filepath: str = None):
        """
        モデルを保存
        """
        if filepath is None:
            filepath = f'/Volumes/FUKUI-SSD01/fx_trade/models/{self.model_name}.h5'
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        モデルを読み込む
        """
        # Keras 3.xのH5形式の互換性問題を回避するため、compile=Falseを指定
        self.model = keras.models.load_model(filepath, compile=False)
        print(f"Model loaded from {filepath}")
        
        return self.model
    
    def predict(self, X):
        """
        予測を実行
        """
        if self.model is None:
            raise ValueError("Model not loaded or built.")
        
        return self.model.predict(X)


if __name__ == "__main__":
    # テスト用
    pass

