"""
Focal Loss実装
クラスウェイトよりも強力な不均衡対策
"""

import tensorflow as tf
import keras
import numpy as np
from typing import Optional, Union

def focal_loss(alpha: Optional[Union[float, list]] = None, 
               gamma: float = 2.0,
               from_logits: bool = False) -> callable:
    """
    Focal Loss実装
    
    Args:
        alpha: クラス重み（float or list）
        gamma: focusing parameter（デフォルト2.0）
        from_logits: logitsから確率を計算するか
    
    Returns:
        focal_loss関数
    """
    def focal_loss_fixed(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Focal Loss計算
        
        Focal Loss = -alpha * (1-p_t)^gamma * log(p_t)
        
        Args:
            y_true: 真のラベル (batch_size, num_classes)
            y_pred: 予測確率 (batch_size, num_classes)
        
        Returns:
            focal loss値
        """
        # logitsから確率に変換（必要に応じて）
        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # 各サンプルの予測確率を取得
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # 真のクラスの予測確率を取得
        y_true_float = tf.cast(y_true, tf.float32)
        p_t = tf.reduce_sum(y_true_float * y_pred, axis=-1)
        
        # alpha重みを適用
        if alpha is not None:
            if isinstance(alpha, (list, tuple, np.ndarray)):
                alpha = tf.constant(alpha, dtype=tf.float32)
                alpha_t = tf.reduce_sum(alpha * y_true_float, axis=-1)
            else:
                alpha_t = alpha
        else:
            alpha_t = 1.0
        
        # Focal Loss計算
        focal_weight = alpha_t * tf.pow(1.0 - p_t, gamma)
        ce = -tf.reduce_sum(y_true_float * tf.math.log(y_pred), axis=-1)
        focal_loss_value = focal_weight * ce
        
        return tf.reduce_mean(focal_loss_value)
    
    return focal_loss_fixed

def categorical_focal_loss(alpha: Optional[Union[float, list]] = None, 
                           gamma: float = 2.0) -> callable:
    """
    カテゴリカルFocal Loss（シンプル版）
    
    Args:
        alpha: クラス重み
        gamma: focusing parameter
    
    Returns:
        focal_loss関数
    """
    def focal_loss_fixed(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # 確率に変換
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # 真のクラスの予測確率
        y_true_float = tf.cast(y_true, tf.float32)
        p_t = tf.reduce_sum(y_true_float * y_pred, axis=-1)
        
        # alpha重みの処理（シンプル版）
        if alpha is not None and isinstance(alpha, (list, tuple, np.ndarray)):
            # クラス重みを適用
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            alpha_t = tf.reduce_sum(alpha_tensor * y_true_float, axis=-1)
        else:
            alpha_t = tf.constant(1.0, dtype=tf.float32)
        
        # Focal Loss計算
        focal_weight = alpha_t * tf.pow(1.0 - p_t, gamma)
        ce = -tf.reduce_sum(y_true_float * tf.math.log(y_pred), axis=-1)
        focal_loss_value = focal_weight * ce
        
        return tf.reduce_mean(focal_loss_value)
    
    return focal_loss_fixed

def binary_focal_loss(alpha: float = 0.25, gamma: float = 2.0) -> callable:
    """
    バイナリFocal Loss
    
    Args:
        alpha: 正例の重み
        gamma: focusing parameter
    
    Returns:
        focal_loss関数
    """
    def focal_loss_fixed(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # 確率に変換
        y_pred = tf.nn.sigmoid(y_pred)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # 真のラベル
        y_true_float = tf.cast(y_true, tf.float32)
        
        # 正例・負例の確率
        p_t = tf.where(tf.equal(y_true_float, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true_float, 1), alpha, 1 - alpha)
        
        # Focal Loss
        focal_weight = alpha_t * tf.pow(1.0 - p_t, gamma)
        ce = -y_true_float * tf.math.log(y_pred) - (1 - y_true_float) * tf.math.log(1 - y_pred)
        focal_loss_value = focal_weight * ce
        
        return tf.reduce_mean(focal_loss_value)
    
    return focal_loss_fixed

# テスト用の関数
def test_focal_loss():
    """Focal Lossのテスト"""
    print("=== Focal Loss テスト ===")
    
    # テストデータ
    batch_size = 100
    num_classes = 3
    
    # 不均衡データをシミュレート
    y_true = np.zeros((batch_size, num_classes))
    y_pred = np.random.random((batch_size, num_classes))
    
    # 90%がクラス0（マジョリティ）
    for i in range(batch_size):
        if i < 90:
            y_true[i, 0] = 1  # マジョリティクラス
        elif i < 95:
            y_true[i, 1] = 1  # 少数クラス1
        else:
            y_true[i, 2] = 1  # 少数クラス2
    
    # 予測を正規化
    y_pred = tf.nn.softmax(y_pred)
    
    # 通常のCross Entropy
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    ce_loss = tf.reduce_mean(ce_loss)
    
    # Focal Loss
    alpha = [0.1, 10.0, 10.0]  # クラス0を抑制、クラス1,2を増強
    focal_loss_fn = categorical_focal_loss(alpha=alpha, gamma=2.0)
    focal_loss_value = focal_loss_fn(y_true, y_pred)
    
    print(f"Cross Entropy Loss: {ce_loss:.4f}")
    print(f"Focal Loss: {focal_loss_value:.4f}")
    print(f"Focal Loss / CE Ratio: {focal_loss_value / ce_loss:.4f}")
    
    return focal_loss_value

if __name__ == "__main__":
    test_focal_loss()
