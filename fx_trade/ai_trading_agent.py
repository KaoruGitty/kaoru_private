"""
AI Trading Agent System
現在のシステムの問題を解決する高度なAIエージェント
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.feature_engineering import FeatureEngineering
from src.model import LSTMModel, TimeSeriesDataPreparator

class MarketRegimeDetector:
    """
    市場環境を検出するエージェント
    """
    
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.regime_history = []
        
    def detect_regime(self, data: pd.DataFrame) -> str:
        """
        現在の市場環境を検出
        
        Returns:
            str: 'bull' (強気), 'bear' (弱気), 'sideways' (横ばい), 'volatile' (高ボラティリティ)
        """
        if len(data) < self.lookback_window:
            return 'unknown'
        
        recent_data = data.tail(self.lookback_window)
        close_prices = recent_data['close']
        
        # 価格トレンド分析
        price_change = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
        
        # ボラティリティ分析
        returns = close_prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年率換算
        
        # トレンド強度
        sma_short = close_prices.rolling(5).mean()
        sma_long = close_prices.rolling(15).mean()
        trend_strength = abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
        
        # 市場環境の判定
        if volatility > 0.3:  # 30%以上の年率ボラティリティ
            regime = 'volatile'
        elif price_change > 0.05:  # 5%以上の上昇
            regime = 'bull'
        elif price_change < -0.05:  # 5%以上の下降
            regime = 'bear'
        else:
            regime = 'sideways'
        
        self.regime_history.append(regime)
        return regime
    
    def get_regime_confidence(self) -> float:
        """市場環境判定の信頼度を計算"""
        if len(self.regime_history) < 5:
            return 0.5
        
        recent_regimes = self.regime_history[-5:]
        most_common = max(set(recent_regimes), key=recent_regimes.count)
        confidence = recent_regimes.count(most_common) / len(recent_regimes)
        
        return confidence

class AdaptiveThresholdAgent:
    """
    市場環境に応じて閾値を動的調整するエージェント
    """
    
    def __init__(self):
        self.base_thresholds = {
            'price_change_threshold': 0.015,  # 1.5%
            'threshold_return': 0.01,         # 1%
            'window': 3                       # 天底検出の窓
        }
        
        self.regime_multipliers = {
            'bull': {'price_change_threshold': 0.8, 'threshold_return': 0.7},
            'bear': {'price_change_threshold': 1.2, 'threshold_return': 1.3},
            'sideways': {'price_change_threshold': 1.0, 'threshold_return': 1.0},
            'volatile': {'price_change_threshold': 1.5, 'threshold_return': 1.5}
        }
    
    def get_adaptive_thresholds(self, regime: str, volatility: float) -> Dict[str, float]:
        """
        市場環境に応じて閾値を調整
        """
        multipliers = self.regime_multipliers.get(regime, self.regime_multipliers['sideways'])
        
        # ボラティリティに応じた追加調整
        vol_multiplier = 1.0 + (volatility - 0.2) * 0.5  # 20%を基準として調整
        
        adaptive_thresholds = {}
        for key, base_value in self.base_thresholds.items():
            if key in multipliers:
                adaptive_thresholds[key] = base_value * multipliers[key] * vol_multiplier
            else:
                adaptive_thresholds[key] = base_value
        
        return adaptive_thresholds

class MultiModelEnsembleAgent:
    """
    複数のモデルを統合するエージェント
    """
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.models = {}
        self.model_weights = {}
        self.performance_history = {}
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """モデルを追加"""
        self.models[name] = model
        self.model_weights[name] = weight
        
    def predict_ensemble(self, X: np.ndarray, model_type: str) -> Dict[str, Any]:
        """
        アンサンブル予測を実行
        """
        predictions = {}
        weighted_predictions = []
        
        for name, model in self.models.items():
            if model_type in name.lower():
                try:
                    if hasattr(model, 'predict'):
                        # Random Forestモデルの場合はverboseパラメータを除外
                        if 'rf_' in name.lower():
                            pred = model.predict(X)
                        else:
                            pred = model.predict(X, verbose=0)
                        predictions[name] = pred
                        
                        # 重み付き予測
                        weight = self.model_weights.get(name, 1.0)
                        weighted_predictions.append(pred * weight)
                except Exception as e:
                    print(f"Warning: Model {name} prediction failed: {e}")
                    continue
        
        if not weighted_predictions:
            return {'ensemble_pred': None, 'individual_predictions': predictions}
        
        # 重み付き平均
        total_weight = sum(self.model_weights.values())
        ensemble_pred = np.sum(weighted_predictions, axis=0) / total_weight
        
        return {
            'ensemble_pred': ensemble_pred,
            'individual_predictions': predictions,
            'model_weights': self.model_weights
        }
    
    def update_model_weights(self, performance_scores: Dict[str, float]):
        """モデルの重みをパフォーマンスに基づいて更新"""
        total_performance = sum(performance_scores.values())
        
        for name, score in performance_scores.items():
            if total_performance > 0:
                self.model_weights[name] = score / total_performance
            else:
                self.model_weights[name] = 1.0 / len(performance_scores)

class RiskManagementAgent:
    """
    リスク管理を担当するエージェント
    """
    
    def __init__(self, max_position_size: float = 0.1, max_drawdown: float = 0.15):
        self.max_position_size = max_position_size  # 最大ポジションサイズ（10%）
        self.max_drawdown = max_drawdown            # 最大ドローダウン（15%）
        self.current_position = 0
        self.portfolio_value = 1000000
        self.peak_value = 1000000
        self.trade_history = []
        
    def assess_risk(self, signal: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        取引シグナルのリスクを評価
        """
        # 現在のドローダウン計算
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        # リスク評価
        risk_assessment = {
            'is_acceptable': True,
            'position_size': 0,
            'risk_score': 0.0,
            'warnings': []
        }
        
        # ドローダウンチェック
        if current_drawdown > self.max_drawdown:
            risk_assessment['is_acceptable'] = False
            risk_assessment['warnings'].append(f"Max drawdown exceeded: {current_drawdown:.2%}")
        
        # ポジションサイズ計算
        if signal.get('action') == 'buy':
            # 予測の信頼度に基づいてポジションサイズを調整
            confidence = signal.get('confidence', 0.5)
            base_size = self.max_position_size * confidence
            
            # ボラティリティ調整
            volatility = signal.get('volatility', 0.2)
            vol_adjustment = max(0.5, 1.0 - volatility)
            
            position_size = base_size * vol_adjustment
            risk_assessment['position_size'] = min(position_size, self.max_position_size)
            
            # リスクスコア計算
            risk_score = (1.0 - confidence) + volatility + current_drawdown
            risk_assessment['risk_score'] = min(risk_score, 1.0)
            
            if risk_score > 0.7:
                risk_assessment['warnings'].append("High risk score detected")
        
        return risk_assessment
    
    def update_portfolio(self, trade_result: Dict[str, Any]):
        """ポートフォリオ状態を更新"""
        self.portfolio_value = trade_result.get('portfolio_value', self.portfolio_value)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.trade_history.append(trade_result)

class IntelligentTradingAgent:
    """
    メインのAIトレーディングエージェント
    """
    
    def __init__(self, ticker: str, lookback: int = 60):
        self.ticker = ticker
        self.lookback = lookback
        
        # サブエージェントを初期化
        self.regime_detector = MarketRegimeDetector()
        self.threshold_agent = AdaptiveThresholdAgent()
        self.ensemble_agent = MultiModelEnsembleAgent(lookback)
        self.risk_agent = RiskManagementAgent()
        
        # データとモデル
        self.data = None
        self.feature_columns = None
        self.scaler = RobustScaler()
        
        # パフォーマンス追跡
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def initialize(self, period: str = "10y"):
        """エージェントを初期化"""
        print("🤖 Initializing AI Trading Agent...")
        
        # データの取得と前処理
        self._load_and_prepare_data(period)
        
        # モデルの学習
        self._train_models()
        
        print("✅ AI Trading Agent initialized successfully!")
    
    def _load_and_prepare_data(self, period: str):
        """データの読み込みと前処理"""
        print("📊 Loading and preparing data...")
        
        # データ取得
        fetcher = DataFetcher(self.ticker, period=period)
        data = fetcher.fetch_data()
        
        # 特徴量エンジニアリング
        engineer = FeatureEngineering(data.copy())
        engineer.add_technical_indicators()
        engineer.detect_peaks_and_bottoms()
        engineer.create_target_labels()
        engineer.create_time_series_features()
        data = engineer.handle_missing_values()
        
        self.data = data
        
        # 特徴量選択
        self._select_features()
        
        print(f"✅ Data prepared: {len(data)} rows, {len(self.feature_columns)} features")
    
    def _select_features(self):
        """特徴量を選択"""
        base_features = ['open', 'high', 'low', 'close', 'volume']
        technical_features = [col for col in self.data.columns if any(
            indicator in col for indicator in [
                'sma', 'ema', 'macd', 'rsi', 'stoch', 'bb', 'atr', 'obv',
                'return', 'volatility', 'volume_sma', 'volume_ratio'
            ]
        )]
        time_features = ['day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter']
        
        self.feature_columns = base_features + technical_features + time_features
        self.feature_columns = [col for col in self.feature_columns if col in self.data.columns]
        
        # ターゲット変数を除外
        exclude_columns = ['next_day_return', 'target_return', 'target_class', 'target_class_mapped',
                           'target_peak_bottom', 'is_peak', 'is_bottom', 'buy_signal']
        self.feature_columns = [col for col in self.feature_columns if col not in exclude_columns]
    
    def _train_models(self):
        """モデルを学習"""
        print("🧠 Training models...")
        
        # データ準備
        required_cols = self.feature_columns + ['target_return', 'target_class', 'target_peak_bottom']
        df_clean = self.data[required_cols].dropna()
        
        # 特徴量の正規化
        X_data = df_clean[self.feature_columns].values
        X_scaled = self.scaler.fit_transform(X_data)
        
        # ターゲットデータ
        y_reg = df_clean['target_return'].values
        y_class = df_clean['target_class'].values
        y_peak_bottom = df_clean['target_peak_bottom'].values
        
        # 時系列データの準備
        preparator = TimeSeriesDataPreparator(lookback=self.lookback)
        X_seq, y_reg_seq = preparator.create_sequences(X_scaled, y_reg)
        _, y_class_seq = preparator.create_sequences(X_scaled, y_class)
        _, y_peak_bottom_seq = preparator.create_sequences(X_scaled, y_peak_bottom)
        
        # データ分割
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train_reg, y_test_reg = y_reg_seq[:split_idx], y_reg_seq[split_idx:]
        y_train_class, y_test_class = y_class_seq[:split_idx], y_class_seq[split_idx:]
        y_train_pb, y_test_pb = y_peak_bottom_seq[:split_idx], y_peak_bottom_seq[split_idx:]
        
        # 分類データのone-hotエンコード
        y_train_class_oh = tf.keras.utils.to_categorical(y_train_class, num_classes=3)
        y_train_pb_oh = tf.keras.utils.to_categorical(y_train_pb, num_classes=3)
        
        # LSTMモデルの学習
        self._train_lstm_models(X_train, y_train_reg, y_train_class_oh, y_train_pb_oh)
        
        # 従来の機械学習モデルも追加
        self._train_traditional_models(X_train, y_train_reg, y_train_class, y_train_pb)
        
        print("✅ Models trained successfully!")
    
    def _train_lstm_models(self, X_train, y_train_reg, y_train_class, y_train_pb):
        """LSTMモデルを学習"""
        # 回帰モデル
        reg_model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]))
        reg_model.build_regression_model(lstm_units=[128, 64], dropout_rate=0.3)
        reg_model.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # 分類モデル（方向）
        class_model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]))
        class_model.build_classification_model(num_classes=3, lstm_units=[128, 64], dropout_rate=0.3)
        class_model.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # 天底検出モデル
        pb_model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]))
        pb_model.build_classification_model(num_classes=3, lstm_units=[128, 64], dropout_rate=0.3)
        pb_model.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # モデルをアンサンブルに追加
        self.ensemble_agent.add_model('lstm_regression', reg_model.model, weight=0.4)
        self.ensemble_agent.add_model('lstm_classification', class_model.model, weight=0.3)
        self.ensemble_agent.add_model('lstm_peak_bottom', pb_model.model, weight=0.3)
    
    def _train_traditional_models(self, X_train, y_train_reg, y_train_class, y_train_pb):
        """従来の機械学習モデルを学習"""
        # 特徴量を2次元に変換（LSTM用の3次元から）
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        
        # ランダムフォレスト
        rf_reg = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_class = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_pb = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 学習
        rf_reg.fit(X_train_2d, (y_train_reg > 0.01).astype(int))  # 1%以上の上昇を予測
        rf_class.fit(X_train_2d, y_train_class)
        rf_pb.fit(X_train_2d, y_train_pb)
        
        # アンサンブルに追加
        self.ensemble_agent.add_model('rf_regression', rf_reg, weight=0.2)
        self.ensemble_agent.add_model('rf_classification', rf_class, weight=0.2)
        self.ensemble_agent.add_model('rf_peak_bottom', rf_pb, weight=0.2)
    
    def generate_trading_signal(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        現在のデータから取引シグナルを生成
        """
        # 市場環境の検出
        regime = self.regime_detector.detect_regime(current_data)
        regime_confidence = self.regime_detector.get_regime_confidence()
        
        # 適応的閾値の取得
        recent_volatility = current_data['close'].pct_change().std() * np.sqrt(252)
        thresholds = self.threshold_agent.get_adaptive_thresholds(regime, recent_volatility)
        
        # 特徴量の準備
        if len(current_data) < self.lookback:
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        # 最新のデータを取得
        recent_data = current_data.tail(self.lookback)
        X = recent_data[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        X_seq = X_scaled.reshape(1, self.lookback, -1)
        
        # アンサンブル予測
        reg_pred = self.ensemble_agent.predict_ensemble(X_seq, 'regression')
        class_pred = self.ensemble_agent.predict_ensemble(X_seq, 'classification')
        pb_pred = self.ensemble_agent.predict_ensemble(X_seq, 'peak_bottom')
        
        # 予測結果の処理
        predicted_return = reg_pred['ensemble_pred'][0][0] if reg_pred['ensemble_pred'] is not None else 0.0
        predicted_direction = np.argmax(class_pred['ensemble_pred'][0]) if class_pred['ensemble_pred'] is not None else 1
        predicted_peak_bottom = np.argmax(pb_pred['ensemble_pred'][0]) if pb_pred['ensemble_pred'] is not None else 0
        
        # 信頼度の計算
        confidence = self._calculate_confidence(reg_pred, class_pred, pb_pred, regime_confidence)
        
        # 取引シグナルの生成
        signal = self._generate_signal(
            predicted_return, predicted_direction, predicted_peak_bottom,
            confidence, thresholds, regime
        )
        
        return signal
    
    def _calculate_confidence(self, reg_pred, class_pred, pb_pred, regime_confidence):
        """予測の信頼度を計算"""
        # 各モデルの予測の一貫性を評価
        reg_consistency = 1.0
        class_consistency = 1.0
        pb_consistency = 1.0
        
        # 個別予測の分散を計算
        if reg_pred['individual_predictions']:
            reg_values = [pred[0][0] for pred in reg_pred['individual_predictions'].values()]
            reg_consistency = 1.0 - np.std(reg_values) / (np.mean(np.abs(reg_values)) + 1e-8)
        
        if class_pred['individual_predictions']:
            class_values = [np.argmax(pred[0]) for pred in class_pred['individual_predictions'].values()]
            class_consistency = len(set(class_values)) == 1  # 全モデルが同じ予測
        
        if pb_pred['individual_predictions']:
            pb_values = [np.argmax(pred[0]) for pred in pb_pred['individual_predictions'].values()]
            pb_consistency = len(set(pb_values)) == 1
        
        # 総合信頼度
        overall_confidence = (reg_consistency + class_consistency + pb_consistency) / 3.0
        overall_confidence *= regime_confidence
        
        return min(overall_confidence, 1.0)
    
    def _generate_signal(self, predicted_return, predicted_direction, predicted_peak_bottom,
                        confidence, thresholds, regime):
        """取引シグナルを生成"""
        signal = {
            'action': 'hold',
            'confidence': confidence,
            'predicted_return': predicted_return,
            'predicted_direction': predicted_return,
            'predicted_peak_bottom': predicted_peak_bottom,
            'regime': regime,
            'thresholds': thresholds,
            'reason': ''
        }
        
        # 買いシグナルの条件
        buy_conditions = [
            predicted_direction == 2 and predicted_return > thresholds['threshold_return'],
            predicted_peak_bottom == 2 and predicted_return > thresholds['threshold_return'] * 0.5,
            predicted_direction == 1 and predicted_return > thresholds['threshold_return'] * 2,
            predicted_peak_bottom == 0 and predicted_return > thresholds['threshold_return'] * 1.5
        ]
        
        if any(buy_conditions) and confidence > 0.6:
            signal['action'] = 'buy'
            signal['reason'] = f"Buy signal: return={predicted_return:.4f}, direction={predicted_direction}, confidence={confidence:.3f}"
        
        # 売りシグナルの条件
        elif predicted_direction == 0 or predicted_return < -thresholds['threshold_return'] * 0.5:
            signal['action'] = 'sell'
            signal['reason'] = f"Sell signal: return={predicted_return:.4f}, direction={predicted_direction}"
        
        return signal
    
    def execute_trade(self, signal: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """取引を実行"""
        # リスク評価
        risk_assessment = self.risk_agent.assess_risk(signal, current_price)
        
        if not risk_assessment['is_acceptable']:
            return {
                'executed': False,
                'reason': f"Risk not acceptable: {', '.join(risk_assessment['warnings'])}"
            }
        
        # 取引実行
        trade_result = {
            'executed': True,
            'action': signal['action'],
            'price': current_price,
            'confidence': signal['confidence'],
            'position_size': risk_assessment['position_size'],
            'risk_score': risk_assessment['risk_score'],
            'timestamp': datetime.now()
        }
        
        # ポートフォリオ更新
        self.risk_agent.update_portfolio(trade_result)
        
        # パフォーマンス更新
        self._update_performance_metrics(trade_result)
        
        return trade_result
    
    def _update_performance_metrics(self, trade_result: Dict[str, Any]):
        """パフォーマンス指標を更新"""
        self.performance_metrics['total_trades'] += 1
        
        if trade_result.get('profit', 0) > 0:
            self.performance_metrics['winning_trades'] += 1
        
        # その他の指標も更新...
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリーを取得"""
        win_rate = (self.performance_metrics['winning_trades'] / 
                   max(self.performance_metrics['total_trades'], 1)) * 100
        
        return {
            'total_trades': self.performance_metrics['total_trades'],
            'win_rate': win_rate,
            'total_return': self.performance_metrics['total_return'],
            'max_drawdown': self.performance_metrics['max_drawdown'],
            'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
            'current_regime': self.regime_detector.regime_history[-1] if self.regime_detector.regime_history else 'unknown',
            'regime_confidence': self.regime_detector.get_regime_confidence()
        }

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("🤖 AI Trading Agent System")
    print("=" * 80)
    
    # AIエージェントを初期化
    agent = IntelligentTradingAgent(ticker="7203.T", lookback=60)
    agent.initialize(period="10y")
    
    # パフォーマンスサマリーを表示
    summary = agent.get_performance_summary()
    print("\n📊 Agent Performance Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 AI Trading Agent System initialized successfully!")
    print("Ready for live trading or backtesting...")

if __name__ == "__main__":
    main()
