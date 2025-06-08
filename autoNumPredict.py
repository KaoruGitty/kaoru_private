import tkinter as tk
from tkinter import Button, Label, Canvas
import PIL.Image
import PIL.ImageDraw
import numpy as np
import sklearn.datasets
import sklearn.svm
import os # ファイルパスの操作に必要

# --- グローバル変数としてモデルをロード ---
# アプリケーション起動時に一度だけモデルを学習させる
digits = sklearn.datasets.load_digits()
clf = sklearn.svm.SVC(gamma=0.001)
clf.fit(digits.data, digits.target)

# --- 描画キャンバスと画像処理用の変数を初期化 ---
# 描画内容を保存するためのPIL Imageオブジェクト
# 描画サイズは、後で8x8に縮小しやすいように大きめにする（例: 256x256）
IMAGE_SIZE = 256
canvas_image = PIL.Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255) # L: グレースケール, 255: 白背景
draw = PIL.ImageDraw.Draw(canvas_image) # このImageオブジェクトに描画するためのDrawオブジェクト

# --- 描画状態を管理する変数 ---
last_x, last_y = None, None # 前回のマウス位置

# --- 関数定義 ---

def ImageToData_from_canvas(pil_image):
    """
    PIL Image オブジェクトをdigitsデータセット形式のNumPy配列に変換する
    """
    # グレースケールは既にconvert("L")で処理済み
    # リサイズ (8x8)
    resized_image = pil_image.resize((8, 8), PIL.Image.Resampling.LANCZOS)
    
    # NumPy配列に変換
    num_image = np.asarray(resized_image, dtype=float)
    
    # 濃淡の反転と正規化 (0-16スケール)
    # 元の画像は白(255)が背景、黒(0)が描画なので、この変換で白(0)が背景、黒(16)が描画になる
    num_image = 16 - np.floor(17 * num_image / 256)
    
    # 1次元に平坦化
    num_image = num_image.flatten()
    
    return num_image

def predict_drawn_digit():
    """
    キャンバスに描かれた数字を予測し、結果を表示する
    """
    global canvas_image # グローバル変数のcanvas_imageにアクセス

    # キャンバスの内容をPIL Imageオブジェクトとして取得 (直接は取得できないので、事前に描画内容をcanvas_imageに保存している)
    # ここでは、canvas_imageが現在の描画内容を常に保持していると仮定
    
    # ImageToData関数でデータ形式を変換
    data = ImageToData_from_canvas(canvas_image)
    
    # 予測を実行
    prediction = clf.predict([data])
    
    # 結果をUIに表示
    prediction_label.config(text=f"予測: {prediction[0]}")

    # デバッグ用に変換後のデータと8x8画像を表示
    # plt.figure(figsize=(2,2))
    # plt.imshow(data.reshape(8,8), cmap='gray', vmin=0, vmax=16)
    # plt.title(f"Processed Image (Pred: {prediction[0]})")
    # plt.axis('off')
    # plt.show()


def clear_canvas():
    """
    キャンバスと内部のPIL Imageオブジェクトをクリアする
    """
    global canvas_image, draw, last_x, last_y
    canvas.delete("all") # Tkinterキャンバスの描画をクリア
    canvas_image = PIL.Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255) # 新しい白い画像で初期化
    draw = PIL.ImageDraw.Draw(canvas_image) # Drawオブジェクトも再作成
    prediction_label.config(text="予測: ") # 予測結果もクリア
    last_x, last_y = None, None # マウス位置もリセット

def start_draw(event):
    """
    マウスボタンが押された時の処理
    """
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw_line(event):
    """
    マウスがドラッグされた時の処理
    """
    global last_x, last_y
    if last_x is not None and last_y is not None:
        # Tkinterキャンバスに線を描画 (太く、黒色)
        canvas.create_line(last_x, last_y, event.x, event.y,
                           width=15, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE) # 太さを調整
        
        # PIL Imageオブジェクトにも同じ線を描画 (太く、黒色)
        # Tkinterの座標(0-256)をそのまま使う
        draw.line([last_x, last_y, event.x, event.y], fill=0, width=15, joint="round") # fill=0で黒
        
        last_x, last_y = event.x, event.y

def stop_draw(event):
    """
    マウスボタンが離された時の処理
    描画が終わった後に予測を実行
    """
    global last_x, last_y
    last_x, last_y = None, None
    predict_drawn_digit() # 描画終了時に予測を実行

# --- GUIのセットアップ ---
root = tk.Tk()
root.title("手書き数字認識アプリ")

# 描画キャンバス
canvas = Canvas(root, bg="white", width=IMAGE_SIZE, height=IMAGE_SIZE, bd=2, relief="groove")
canvas.pack(pady=10)

# マウスイベントのバインド
canvas.bind("<Button-1>", start_draw) # マウス左ボタン押し下げ
canvas.bind("<B1-Motion>", draw_line) # マウス左ボタン押しながら移動
canvas.bind("<ButtonRelease-1>", stop_draw) # マウス左ボタン離す

# 予測結果を表示するラベル
prediction_label = Label(root, text="予測: ", font=("Helvetica", 24))
prediction_label.pack(pady=5)

# クリアボタン
clear_button = Button(root, text="クリア", command=clear_canvas)
clear_button.pack(pady=5)

# アプリケーションの実行
root.mainloop()