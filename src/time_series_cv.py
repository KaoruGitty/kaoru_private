"""
時系列クロスバリデーション実装
データ分割の根本改善
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Generator
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesCrossValidator:
    """
    時系列クロスバリデーション
    
    時系列データの特性を考慮した分割方法
    """
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2, 
                 gap: int = 0, max_train_size: int = None):
        """
        Args:
            n_splits: 分割数
            test_size: テストデータの割合
            gap: 学習とテストの間のギャップ（日数）
            max_train_size: 最大学習データサイズ
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size
    
    def split(self, data: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        時系列データを分割
        
        Args:
            data: 時系列データフレーム
        
        Yields:
            (train_indices, test_indices)
        """
        n_samples = len(data)
        test_size_samples = int(n_samples * self.test_size)
        
        # 各foldのテスト期間を計算
        for i in range(self.n_splits):
            # テスト期間の開始位置
            test_start = n_samples - test_size_samples * (self.n_splits - i)
            test_end = test_start + test_size_samples
            
            # ギャップを考慮
            train_end = test_start - self.gap
            
            # 学習期間の開始位置
            if self.max_train_size:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0
            
            # インデックスを生成
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            # 有効なインデックスのみを返す
            train_indices = train_indices[train_indices >= 0]
            test_indices = test_indices[test_indices < n_samples]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

class MarketEnvironmentSplitter:
    """
    市場環境を考慮した分割
    
    異なる市場環境（上昇相場、下降相場、ボラティリティ高）で学習・テスト
    """
    
    def __init__(self):
        self.bull_market_years = [2017, 2019, 2021, 2024]  # 上昇相場
        self.bear_market_years = [2015, 2018, 2020, 2022]  # 下降相場
        self.volatile_years = [2016, 2020, 2023]           # ボラティリティ高
    
    def split_by_market_environment(self, data: pd.DataFrame) -> dict:
        """
        市場環境別にデータを分割
        
        Args:
            data: 時系列データフレーム
        
        Returns:
            市場環境別のデータ辞書
        """
        data['year'] = data.index.year
        
        splits = {
            'bull_market': data[data['year'].isin(self.bull_market_years)],
            'bear_market': data[data['year'].isin(self.bear_market_years)],
            'volatile_market': data[data['year'].isin(self.volatile_years)]
        }
        
        return splits
    
    def get_environment_splits(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        市場環境別の学習・テスト分割を取得
        
        Args:
            data: 時系列データフレーム
        
        Returns:
            [(train_indices, test_indices), ...]
        """
        splits = self.split_by_market_environment(data)
        results = []
        
        for env_name, env_data in splits.items():
            if len(env_data) > 100:  # 十分なデータがある場合のみ
                # 環境内で時系列分割
                n_samples = len(env_data)
                test_size = int(n_samples * 0.3)  # 30%をテストに
                
                train_indices = env_data.index[:n_samples - test_size]
                test_indices = env_data.index[n_samples - test_size:]
                
                # 元のデータフレームのインデックスに変換
                train_mask = data.index.isin(train_indices)
                test_mask = data.index.isin(test_indices)
                
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                
                if len(train_idx) > 0 and len(test_idx) > 0:
                    results.append((train_idx, test_idx))
        
        return results

class RollingWindowValidator:
    """
    ローリングウィンドウ検証
    
    固定期間の学習データでローリング予測
    """
    
    def __init__(self, train_window: int = 1000, test_window: int = 200, 
                 step_size: int = 100):
        """
        Args:
            train_window: 学習ウィンドウサイズ（日数）
            test_window: テストウィンドウサイズ（日数）
            step_size: ステップサイズ（日数）
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
    
    def split(self, data: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        ローリングウィンドウで分割
        
        Args:
            data: 時系列データフレーム
        
        Yields:
            (train_indices, test_indices)
        """
        n_samples = len(data)
        
        start_idx = 0
        while start_idx + self.train_window + self.test_window < n_samples:
            train_start = start_idx
            train_end = train_start + self.train_window
            test_start = train_end
            test_end = test_start + self.test_window
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
            
            start_idx += self.step_size

def analyze_data_splits(data: pd.DataFrame, splits: List[Tuple[np.ndarray, np.ndarray]], 
                       split_names: List[str] = None):
    """
    データ分割の分析
    
    Args:
        data: 時系列データフレーム
        splits: 分割結果のリスト
        split_names: 分割の名前リスト
    """
    if split_names is None:
        split_names = [f"Split {i+1}" for i in range(len(splits))]
    
    print("=== データ分割分析 ===")
    
    for i, (train_idx, test_idx) in enumerate(splits):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        print(f"\n{split_names[i]}:")
        print(f"  学習期間: {train_data.index[0]} ～ {train_data.index[-1]}")
        print(f"  テスト期間: {test_data.index[0]} ～ {test_data.index[-1]}")
        print(f"  学習サンプル数: {len(train_data)}")
        print(f"  テストサンプル数: {len(test_data)}")
        
        # 年別分布
        train_years = train_data.index.year.value_counts().sort_index()
        test_years = test_data.index.year.value_counts().sort_index()
        
        print(f"  学習期間の年: {list(train_years.index)}")
        print(f"  テスト期間の年: {list(test_years.index)}")
        
        # クラス分布（ターゲットがある場合）
        if 'target_class' in train_data.columns:
            train_class_dist = train_data['target_class'].value_counts().sort_index()
            test_class_dist = test_data['target_class'].value_counts().sort_index()
            
            print(f"  学習期間のクラス分布:")
            for class_val, count in train_class_dist.items():
                print(f"    クラス{class_val}: {count}個 ({count/len(train_data)*100:.1f}%)")
            
            print(f"  テスト期間のクラス分布:")
            for class_val, count in test_class_dist.items():
                print(f"    クラス{class_val}: {count}個 ({count/len(test_data)*100:.1f}%)")

def test_time_series_cv():
    """時系列CVのテスト"""
    print("=== 時系列CV テスト ===")
    
    # テストデータ生成
    dates = pd.date_range('2015-01-01', '2025-01-01', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(len(dates)),
        'target_class': np.random.choice([0, 1, 2], len(dates), p=[0.7, 0.2, 0.1])
    })
    data.set_index('date', inplace=True)
    
    print(f"総データ数: {len(data)}")
    print(f"期間: {data.index[0]} ～ {data.index[-1]}")
    
    # 時系列CV
    tscv = TimeSeriesCrossValidator(n_splits=5, test_size=0.2)
    splits = list(tscv.split(data))
    
    analyze_data_splits(data, splits, [f"TS-CV {i+1}" for i in range(len(splits))])
    
    # 市場環境分割
    mes = MarketEnvironmentSplitter()
    env_splits = mes.get_environment_splits(data)
    
    analyze_data_splits(data, env_splits, ["Bull Market", "Bear Market", "Volatile Market"])
    
    # ローリングウィンドウ
    rwv = RollingWindowValidator(train_window=500, test_window=100, step_size=50)
    rolling_splits = list(rwv.split(data))
    
    print(f"\nローリングウィンドウ分割数: {len(rolling_splits)}")
    if rolling_splits:
        analyze_data_splits(data, rolling_splits[:3], [f"Rolling {i+1}" for i in range(3)])

if __name__ == "__main__":
    test_time_series_cv()
