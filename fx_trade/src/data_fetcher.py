"""
Yahoo!ファイナンスから株価データを取得するモジュール
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os


class DataFetcher:
    def __init__(self, ticker: str, start_date: str = None, end_date: str = None, period: str = "5y"):
        """
        Args:
            ticker: 銘柄コード（例: "7203.T" for トヨタ自動車）
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
            period: 期間（start_date/end_dateが指定されていない場合）
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.period = period
        self.data = None
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Yahoo!ファイナンスからデータを取得
        """
        print(f"Fetching data for {self.ticker}...")
        
        if self.start_date:
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        else:
            data = yf.download(self.ticker, period=self.period, progress=False)
        
        if data.empty:
            raise ValueError(f"No data fetched for {self.ticker}")
        
        # MultiIndexの場合は最初のレベルのみを使用
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # カラム名を小文字に変換
        data.columns = data.columns.str.lower()
        
        self.data = data
        print(f"Data fetched: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
        
        return data
    
    def save_data(self, filepath: str):
        """
        データをCSVファイルとして保存
        """
        if self.data is None:
            raise ValueError("No data to save. Please fetch data first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.data.to_csv(filepath)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        保存されたCSVファイルからデータを読み込む
        """
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Data loaded from {filepath}: {len(self.data)} rows")
        return self.data


if __name__ == "__main__":
    # テスト用
    fetcher = DataFetcher("7203.T", period="2y")  # トヨタ自動車
    data = fetcher.fetch_data()
    print(data.head())
    print(data.tail())

