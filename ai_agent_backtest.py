"""
AI Trading Agent Backtest System
AIエージェントのバックテストと評価システム
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from ai_trading_agent import IntelligentTradingAgent

class AgentBacktester:
    """
    AIエージェントのバックテストシステム
    """
    
    def __init__(self, agent: IntelligentTradingAgent, initial_capital: float = 1000000):
        self.agent = agent
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 0: なし, 1: 買いポジション
        self.shares = 0
        self.trades = []
        self.portfolio_values = [initial_capital]
        self.daily_returns = []
        
    def run_backtest(self, test_data: pd.DataFrame, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        バックテストを実行
        """
        print("🚀 Starting AI Agent Backtest...")
        
        # テスト期間の設定
        if start_date:
            test_data = test_data[test_data.index >= start_date]
        if end_date:
            test_data = test_data[test_data.index <= end_date]
        
        print(f"📅 Test period: {test_data.index[0]} to {test_data.index[-1]}")
        print(f"📊 Test data length: {len(test_data)} days")
        
        # バックテスト実行
        for i in range(len(test_data)):
            current_date = test_data.index[i]
            current_data = test_data.iloc[:i+1]  # 現在時点までのデータ
            
            # 取引シグナルを生成
            signal = self.agent.generate_trading_signal(current_data)
            
            # 現在の価格
            current_price = test_data.iloc[i]['close']
            
            # 取引を実行
            trade_result = self._execute_trade_logic(signal, current_price, current_date)
            
            # ポートフォリオ価値を更新
            self._update_portfolio_value(current_price)
            
            # 日次リターンを記録
            if len(self.portfolio_values) > 1:
                daily_return = (self.portfolio_values[-1] - self.portfolio_values[-2]) / self.portfolio_values[-2]
                self.daily_returns.append(daily_return)
        
        # 最終ポジションを清算
        if self.position == 1:
            final_price = test_data.iloc[-1]['close']
            self.capital = self.shares * final_price
            self.trades.append({
                'date': test_data.index[-1],
                'action': 'sell',
                'price': final_price,
                'shares': self.shares,
                'capital': self.capital,
                'reason': 'Final liquidation'
            })
            self.position = 0
            self.shares = 0
        
        # 結果を計算
        results = self._calculate_results(test_data)
        
        print("✅ Backtest completed!")
        return results
    
    def _execute_trade_logic(self, signal: Dict[str, Any], current_price: float, current_date) -> Dict[str, Any]:
        """取引ロジックを実行"""
        trade_result = {'executed': False}
        
        if signal['action'] == 'buy' and self.position == 0:
            # 買いシグナル
            position_size = signal.get('confidence', 0.5) * 0.1  # 信頼度に基づくポジションサイズ
            self.shares = (self.capital * position_size) / current_price
            self.position = 1
            
            self.trades.append({
                'date': current_date,
                'action': 'buy',
                'price': current_price,
                'shares': self.shares,
                'capital': self.capital,
                'confidence': signal['confidence'],
                'reason': signal['reason']
            })
            
            trade_result = {
                'executed': True,
                'action': 'buy',
                'price': current_price,
                'shares': self.shares,
                'confidence': signal['confidence']
            }
            
            print(f"🟢 BUY  {current_date.date()}: Price={current_price:.2f}, Confidence={signal['confidence']:.3f}")
        
        elif signal['action'] == 'sell' and self.position == 1:
            # 売りシグナル
            self.capital = self.shares * current_price
            
            self.trades.append({
                'date': current_date,
                'action': 'sell',
                'price': current_price,
                'shares': self.shares,
                'capital': self.capital,
                'reason': signal['reason']
            })
            
            trade_result = {
                'executed': True,
                'action': 'sell',
                'price': current_price,
                'capital': self.capital
            }
            
            print(f"🔴 SELL {current_date.date()}: Price={current_price:.2f}, Capital={self.capital:.2f}")
            
            self.position = 0
            self.shares = 0
        
        return trade_result
    
    def _update_portfolio_value(self, current_price: float):
        """ポートフォリオ価値を更新"""
        if self.position == 1:
            portfolio_value = self.shares * current_price
        else:
            portfolio_value = self.capital
        
        self.portfolio_values.append(portfolio_value)
    
    def _calculate_results(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """バックテスト結果を計算"""
        final_capital = self.capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
        # Buy & Hold リターン
        buy_hold_return = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
        
        # 取引統計
        buy_trades = [t for t in self.trades if t['action'] == 'buy']
        sell_trades = [t for t in self.trades if t['action'] == 'sell']
        
        # 勝率計算
        profits = []
        for i in range(1, len(self.trades), 2):
            if i < len(self.trades) and self.trades[i]['action'] == 'sell':
                profit = self.trades[i]['capital'] - self.trades[i-1]['capital']
                profits.append(profit)
        
        win_rate = 0
        average_profit = 0
        average_loss = 0
        profit_factor = 0
        
        if profits:
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p < 0]
            
            win_rate = len(wins) / len(profits) * 100 if profits else 0
            average_profit = np.mean(wins) if wins else 0
            average_loss = np.mean(losses) if losses else 0
            
            if losses and sum(losses) != 0:
                profit_factor = abs(sum(wins) / sum(losses)) if wins else 0
        
        # ドローダウン計算
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        max_drawdown = np.min(drawdown)
        
        # シャープレシオ
        if len(self.daily_returns) > 1:
            sharpe_ratio = np.mean(self.daily_returns) / np.std(self.daily_returns) * np.sqrt(252) if np.std(self.daily_returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # エージェント固有の指標
        agent_performance = self.agent.get_performance_summary()
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'buy_and_hold_return': buy_hold_return,
            'outperformance': total_return - buy_hold_return,
            'total_trades': len(buy_trades),
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'average_profit': average_profit,
            'average_loss': average_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'agent_performance': agent_performance
        }
        
        return results
    
    def plot_results(self, test_data: pd.DataFrame, results: Dict[str, Any], save_path: str = None):
        """結果を可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI Trading Agent Backtest Results', fontsize=16)
        
        # 1. ポートフォリオ価値の推移
        portfolio_values = results['portfolio_values']
        if len(portfolio_values) > len(test_data):
            portfolio_values = portfolio_values[:len(test_data)]
        elif len(portfolio_values) < len(test_data):
            # ポートフォリオ価値が少ない場合は最後の値で埋める
            last_value = portfolio_values[-1] if portfolio_values else self.initial_capital
            portfolio_values.extend([last_value] * (len(test_data) - len(portfolio_values)))
        
        axes[0, 0].plot(test_data.index, portfolio_values)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value (¥)')
        axes[0, 0].grid(True)
        
        # Buy & Holdとの比較
        buy_hold_values = []
        initial_price = test_data.iloc[0]['close']
        for price in test_data['close']:
            buy_hold_values.append(self.initial_capital * (price / initial_price))
        
        axes[0, 0].plot(test_data.index, buy_hold_values, label='Buy & Hold', alpha=0.7)
        axes[0, 0].legend()
        
        # 2. 取引シグナル
        buy_dates = [t['date'] for t in results['trades'] if t['action'] == 'buy']
        sell_dates = [t['date'] for t in results['trades'] if t['action'] == 'sell']
        
        axes[0, 1].plot(test_data.index, test_data['close'], label='Price', alpha=0.7)
        axes[0, 1].scatter(buy_dates, [test_data.loc[d, 'close'] for d in buy_dates], 
                          color='green', marker='^', s=100, label='Buy Signals')
        axes[0, 1].scatter(sell_dates, [test_data.loc[d, 'close'] for d in sell_dates], 
                          color='red', marker='v', s=100, label='Sell Signals')
        axes[0, 1].set_title('Trading Signals')
        axes[0, 1].set_ylabel('Price (¥)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. パフォーマンス指標
        metrics = ['Total Return', 'Buy & Hold Return', 'Outperformance', 'Win Rate', 'Sharpe Ratio']
        values = [
            results['total_return'],
            results['buy_and_hold_return'],
            results['outperformance'],
            results['win_rate'],
            results['sharpe_ratio']
        ]
        
        bars = axes[1, 0].bar(metrics, values)
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # バーの色を設定
        colors = ['green' if v > 0 else 'red' for v in values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 4. ドローダウン
        portfolio_values = np.array(results['portfolio_values'])
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        
        axes[1, 1].fill_between(test_data.index[:len(drawdown)], drawdown, 0, 
                               color='red', alpha=0.3, label='Drawdown')
        axes[1, 1].plot(test_data.index[:len(drawdown)], drawdown, color='red')
        axes[1, 1].set_title('Drawdown Over Time')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Results plot saved to {save_path}")
        
        plt.show()

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("🤖 AI Trading Agent Backtest System")
    print("=" * 80)
    
    # AIエージェントを初期化
    print("🚀 Initializing AI Trading Agent...")
    agent = IntelligentTradingAgent(ticker="7203.T", lookback=60)
    agent.initialize(period="10y")
    
    # バックテスターを初期化
    backtester = AgentBacktester(agent, initial_capital=1000000)
    
    # テストデータを準備
    test_data = agent.data.tail(1000)  # 最新1000日をテスト
    
    # バックテストを実行
    print("\n🚀 Running backtest...")
    results = backtester.run_backtest(test_data)
    
    # 結果を表示
    print("\n" + "=" * 80)
    print("📈 AI Agent Backtest Results")
    print("=" * 80)
    print(f"💰 Initial Capital: ¥{results['initial_capital']:,}")
    print(f"📈 Final Capital: ¥{results['final_capital']:,}")
    print(f"📊 Total Return: {results['total_return']:.2f}%")
    print(f"📉 Buy & Hold Return: {results['buy_and_hold_return']:.2f}%")
    print(f"🚀 Outperformance: {results['outperformance']:.2f}%")
    print(f"🔄 Total Trades: {results['total_trades']}")
    print(f"⬇️ Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"✅ Win Rate: {results['win_rate']:.2f}%")
    print(f"📈 Average Profit: ¥{results['average_profit']:.2f}")
    print(f"📉 Average Loss: ¥{results['average_loss']:.2f}")
    print(f"⚖️ Profit Factor: {results['profit_factor']:.2f}")
    print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    # エージェント固有の指標
    print("\n🤖 Agent Performance:")
    agent_perf = results['agent_performance']
    for key, value in agent_perf.items():
        print(f"  {key}: {value}")
    
    # 結果を可視化
    print("\n📊 Generating visualization...")
    backtester.plot_results(test_data, results, save_path='data/ai_agent_backtest_results.png')
    
    print("\n🎉 AI Agent Backtest completed successfully!")

if __name__ == "__main__":
    main()
