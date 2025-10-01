"""
テクニカル指標の計算と特徴量エンジニアリング
"""
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureEngineering:
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: OHLCVデータ（Open, High, Low, Close, Volume）
        """
        self.data = data.copy()
        self.scaler = StandardScaler()
        
    def add_technical_indicators(self):
        """
        各種テクニカル指標を追加
        """
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        volume = self.data['volume']
        
        # 移動平均線
        for period in [5, 10, 20, 50, 75, 200]:
            sma = SMAIndicator(close=close, window=period)
            self.data[f'sma_{period}'] = sma.sma_indicator()
            
            # 移動平均乖離率
            self.data[f'sma_{period}_divergence'] = (close - self.data[f'sma_{period}']) / self.data[f'sma_{period}'] * 100
        
        # 指数移動平均
        for period in [12, 26]:
            ema = EMAIndicator(close=close, window=period)
            self.data[f'ema_{period}'] = ema.ema_indicator()
        
        # MACD
        macd = MACD(close=close)
        self.data['macd'] = macd.macd()
        self.data['macd_signal'] = macd.macd_signal()
        self.data['macd_diff'] = macd.macd_diff()
        
        # RSI
        for period in [9, 14]:
            rsi = RSIIndicator(close=close, window=period)
            self.data[f'rsi_{period}'] = rsi.rsi()
        
        # ストキャスティクス
        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        self.data['stoch_k'] = stoch.stoch()
        self.data['stoch_d'] = stoch.stoch_signal()
        
        # ボリンジャーバンド
        for period in [20]:
            bb = BollingerBands(close=close, window=period, window_dev=2)
            self.data[f'bb_{period}_high'] = bb.bollinger_hband()
            self.data[f'bb_{period}_low'] = bb.bollinger_lband()
            self.data[f'bb_{period}_mid'] = bb.bollinger_mavg()
            self.data[f'bb_{period}_width'] = bb.bollinger_wband()
            self.data[f'bb_{period}_pband'] = bb.bollinger_pband()
        
        # ATR（Average True Range）
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        self.data['atr'] = atr.average_true_range()
        
        # OBV（On Balance Volume）
        obv = OnBalanceVolumeIndicator(close=close, volume=volume)
        self.data['obv'] = obv.on_balance_volume()
        
        # 価格の変化率
        for period in [1, 3, 5, 10]:
            self.data[f'return_{period}d'] = close.pct_change(period)
            
        # ボラティリティ
        for period in [5, 10, 20]:
            self.data[f'volatility_{period}d'] = close.pct_change().rolling(window=period).std()
        
        # 出来高の移動平均
        for period in [5, 20]:
            self.data[f'volume_sma_{period}'] = volume.rolling(window=period).mean()
            self.data[f'volume_ratio_{period}'] = volume / self.data[f'volume_sma_{period}']
        
        print(f"Technical indicators added. Total features: {len(self.data.columns)}")
        
        return self.data
    
    def detect_peaks_and_bottoms(self, window: int = 5):
        """
        天井（ピーク）と底（ボトム）を検出
        
        Args:
            window: 前後何日間を確認するか
        """
        close = self.data['close']
        
        # ローカルマキシマ（天井）
        self.data['is_peak'] = 0
        for i in range(window, len(close) - window):
            if close.iloc[i] == close.iloc[i-window:i+window+1].max():
                self.data.iloc[i, self.data.columns.get_loc('is_peak')] = 1
        
        # ローカルミニマ（底）
        self.data['is_bottom'] = 0
        for i in range(window, len(close) - window):
            if close.iloc[i] == close.iloc[i-window:i+window+1].min():
                self.data.iloc[i, self.data.columns.get_loc('is_bottom')] = 1
        
        print(f"Peaks detected: {self.data['is_peak'].sum()}")
        print(f"Bottoms detected: {self.data['is_bottom'].sum()}")
        
        return self.data
    
    def create_buy_signals(self, threshold_return: float = 0.02):
        """
        買いシグナルを生成
        底値で、その後上昇するポイント
        
        Args:
            threshold_return: 何%以上の上昇を買いシグナルとするか
        """
        close = self.data['close']
        
        # 次の日に一定以上上昇するポイント
        self.data['next_day_return'] = close.pct_change(1).shift(-1)
        
        # 買いシグナル: 底値で、次の日に上昇
        self.data['buy_signal'] = 0
        buy_condition = (self.data['is_bottom'] == 1) & (self.data['next_day_return'] > threshold_return)
        self.data.loc[buy_condition, 'buy_signal'] = 1
        
        print(f"Buy signals generated: {self.data['buy_signal'].sum()}")
        
        return self.data
    
    def create_target_labels(self, horizon: int = 5, price_change_threshold: float = 0.03):
        """
        学習用の正解ラベルを作成
        
        Args:
            horizon: 何日先の価格を予測するか
            price_change_threshold: 分類の閾値
        """
        close = self.data['close']
        
        # 回帰用: N日後の価格変化率
        self.data['target_return'] = close.pct_change(horizon).shift(-horizon)
        
        # 分類用: 上昇(1)、横ばい(0)、下降(-1)
        self.data['target_class'] = 0
        self.data.loc[self.data['target_return'] > price_change_threshold, 'target_class'] = 1
        self.data.loc[self.data['target_return'] < -price_change_threshold, 'target_class'] = -1
        
        # 天底分類用
        self.data['target_peak_bottom'] = 0  # 通常
        self.data.loc[self.data['is_peak'] == 1, 'target_peak_bottom'] = 1  # 天井
        self.data.loc[self.data['is_bottom'] == 1, 'target_peak_bottom'] = 2  # 底
        
        print(f"Target labels created.")
        print(f"  Price increase: {(self.data['target_class'] == 1).sum()}")
        print(f"  Price decrease: {(self.data['target_class'] == -1).sum()}")
        print(f"  Price neutral: {(self.data['target_class'] == 0).sum()}")
        
        return self.data
    
    def handle_missing_values(self):
        """
        欠損値を補完
        """
        print(f"Missing values before: {self.data.isnull().sum().sum()}")
        
        # 前方補完 → 後方補完
        self.data = self.data.ffill()
        self.data = self.data.bfill()
        
        # まだ残っている場合は0埋め
        self.data.fillna(0, inplace=True)
        
        print(f"Missing values after: {self.data.isnull().sum().sum()}")
        
        return self.data
    
    def create_time_series_features(self):
        """
        時系列特徴量を作成（曜日、月、四半期など）
        """
        # 日付インデックスから特徴量を抽出
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['day_of_month'] = self.data.index.day
        self.data['week_of_year'] = self.data.index.isocalendar().week
        self.data['month'] = self.data.index.month
        self.data['quarter'] = self.data.index.quarter
        
        print(f"Time series features added.")
        
        return self.data
    
    def normalize_features(self, feature_columns: list):
        """
        特徴量を正規化
        
        Args:
            feature_columns: 正規化する特徴量のリスト
        """
        self.data[feature_columns] = self.scaler.fit_transform(self.data[feature_columns])
        
        print(f"Features normalized: {len(feature_columns)} columns")
        
        return self.data
    
    def visualize_signals(self, start_idx: int = 0, end_idx: int = 500):
        """
        買いシグナルと天底を可視化
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        data_slice = self.data.iloc[start_idx:end_idx]
        
        # 株価と移動平均線
        axes[0].plot(data_slice.index, data_slice['close'], label='Close', color='black')
        if 'sma_20' in data_slice.columns:
            axes[0].plot(data_slice.index, data_slice['sma_20'], label='SMA 20', alpha=0.7)
            axes[0].plot(data_slice.index, data_slice['sma_50'], label='SMA 50', alpha=0.7)
        
        # 天底をマーク
        peaks = data_slice[data_slice['is_peak'] == 1]
        bottoms = data_slice[data_slice['is_bottom'] == 1]
        axes[0].scatter(peaks.index, peaks['close'], color='red', marker='v', s=100, label='Peak', zorder=5)
        axes[0].scatter(bottoms.index, bottoms['close'], color='green', marker='^', s=100, label='Bottom', zorder=5)
        
        # 買いシグナルをマーク
        buy_signals = data_slice[data_slice['buy_signal'] == 1]
        axes[0].scatter(buy_signals.index, buy_signals['close'], color='blue', marker='*', s=200, label='Buy Signal', zorder=6)
        
        axes[0].set_ylabel('Price')
        axes[0].set_title('Stock Price with Peaks, Bottoms, and Buy Signals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        if 'rsi_14' in data_slice.columns:
            axes[1].plot(data_slice.index, data_slice['rsi_14'], label='RSI 14', color='purple')
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # MACD
        if 'macd' in data_slice.columns:
            axes[2].plot(data_slice.index, data_slice['macd'], label='MACD', color='blue')
            axes[2].plot(data_slice.index, data_slice['macd_signal'], label='Signal', color='red')
            axes[2].bar(data_slice.index, data_slice['macd_diff'], label='Histogram', alpha=0.3)
            axes[2].set_ylabel('MACD')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.xlabel('Date')
        plt.tight_layout()
        plt.savefig('/Volumes/FUKUI-SSD01/fx_trade/data/signals_visualization.png', dpi=150)
        print("Visualization saved to data/signals_visualization.png")
        plt.close()


if __name__ == "__main__":
    # テスト用
    pass

