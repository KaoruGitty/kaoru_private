#!/usr/bin/env python3
"""
改善版モデル用バックテストスクリプト
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

# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backtest import Backtester
from src.model import LSTMModel

def main():
    print("=" * 80)
    print("改善版モデル バックテスト実行")
    print("=" * 80)
    
    # 改善版データを読み込み
    print("\n📊 改善版データを読み込み中...")
    data_path = "data/processed_data_improved.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ データファイルが見つかりません: {data_path}")
        print("先に改善版の学習を実行してください。")
        return
    
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✅ データ読み込み完了: {len(data)}行")
    print(f"   期間: {data.index[0]} ～ {data.index[-1]}")
    
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
        lookback=60
    )
    
    # 改善版モデルを読み込み
    print("\n📂 改善版モデルを読み込み中...")
    
    # モデルファイルの存在確認
    model_files = {
        'regression': 'models/price_regression_best.h5',  # 回帰モデルは元のものを使用
        'classification': 'models/direction_classification_improved_best.h5',
        'peak_bottom': 'models/peak_bottom_detection_improved_best.h5'
    }
    
    missing_models = []
    for model_type, filepath in model_files.items():
        if not os.path.exists(filepath):
            missing_models.append(f"{model_type}: {filepath}")
    
    if missing_models:
        print("❌ 以下のモデルファイルが見つかりません:")
        for missing in missing_models:
            print(f"   - {missing}")
        print("\n先に改善版の学習を完了してください。")
        return
    
    # モデルを読み込み
    try:
        backtester.load_models()
        print("✅ モデル読み込み完了")
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return
    
    # バックテストを実行
    print("\n🚀 バックテストを実行中...")
    try:
        results = backtester.run_backtest(initial_capital=1_000_000, test_size=0.2)
        
        print("\n" + "=" * 80)
        print("📈 バックテスト結果")
        print("=" * 80)
        
        # 結果を表示
        print(f"\n💰 初期資本: ¥{results['initial_capital']:,}")
        print(f"💰 最終資本: ¥{results['final_capital']:,}")
        print(f"📊 総リターン: {results['total_return']:.2%}")
        print(f"📊 年率リターン: {results['annual_return']:.2%}")
        print(f"📊 最大ドローダウン: {results['max_drawdown']:.2%}")
        print(f"📊 シャープレシオ: {results['sharpe_ratio']:.3f}")
        print(f"📊 勝率: {results['win_rate']:.2%}")
        print(f"📊 総取引数: {results['total_trades']}回")
        print(f"📊 平均リターン/取引: {results['avg_return_per_trade']:.2%}")
        
        # Buy & Holdとの比較
        buy_hold_return = results['buy_hold_return']
        print(f"\n📊 Buy & Hold リターン: {buy_hold_return:.2%}")
        
        if results['total_return'] > buy_hold_return:
            excess_return = results['total_return'] - buy_hold_return
            print(f"🎉 AI戦略がBuy & Holdを {excess_return:.2%} 上回りました！")
        else:
            underperformance = buy_hold_return - results['total_return']
            print(f"⚠️ AI戦略がBuy & Holdを {underperformance:.2%} 下回りました")
        
        # 取引履歴の詳細
        if results['total_trades'] > 0:
            print(f"\n📋 取引履歴（最初の10件）:")
            trades_df = results['trades']
            print(trades_df.head(10).to_string(index=False))
            
            # 月別パフォーマンス
            if 'monthly_returns' in results:
                print(f"\n📅 月別リターン:")
                monthly_df = results['monthly_returns']
                print(monthly_df.head(12).to_string(index=True))
        
        # 結果をCSVに保存
        results_file = "data/backtest_results_improved.csv"
        trades_df = results['trades']
        trades_df.to_csv(results_file, index=False)
        print(f"\n💾 取引履歴を保存: {results_file}")
        
        # 可視化
        print(f"\n📊 結果を可視化中...")
        backtester.plot_results()
        
        print(f"\n✅ バックテスト完了！")
        print(f"   結果画像: data/backtest_results.png")
        print(f"   取引履歴: data/backtest_results_improved.csv")
        
    except Exception as e:
        print(f"❌ バックテスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 元のモデルとの比較
    print("\n" + "=" * 80)
    print("📊 元モデル vs 改善版モデル 比較")
    print("=" * 80)
    
    # 元のバックテスト結果を読み込み（存在する場合）
    original_results_file = "data/backtest_results.csv"
    if os.path.exists(original_results_file):
        try:
            original_trades = pd.read_csv(original_results_file)
            original_trades_count = len(original_trades)
            print(f"\n📊 元モデル:")
            print(f"   総取引数: {original_trades_count}回")
            
            if original_trades_count == 0:
                print(f"   ❌ 取引が発生しませんでした（マジョリティバイアス）")
            else:
                original_total_return = original_trades['total_return'].iloc[-1] if 'total_return' in original_trades.columns else 0
                print(f"   総リターン: {original_total_return:.2%}")
            
            print(f"\n📊 改善版モデル:")
            print(f"   総取引数: {results['total_trades']}回")
            print(f"   総リターン: {results['total_return']:.2%}")
            
            if results['total_trades'] > original_trades_count:
                improvement = results['total_trades'] - original_trades_count
                print(f"\n🎉 改善版で {improvement}回 の追加取引が発生しました！")
            
        except Exception as e:
            print(f"⚠️ 元モデルとの比較でエラー: {e}")
    else:
        print(f"\n📊 改善版モデル:")
        print(f"   総取引数: {results['total_trades']}回")
        print(f"   総リターン: {results['total_return']:.2%}")
        print(f"   年率リターン: {results['annual_return']:.2%}")
        print(f"   最大ドローダウン: {results['max_drawdown']:.2%}")
    
    print("\n" + "=" * 80)
    print("🎯 改善版バックテスト完了！")
    print("=" * 80)

if __name__ == "__main__":
    main()
