import cv2
import numpy as np

# トラックバーのコールバック関数
def on_trackbar(val):
    pass

# オープニング処理とクロージング処理
def apply_morphological_operations(gray, kernel):
    # オープニング処理（膨張 -> 収縮）
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # クロージング処理（収縮 -> 膨張）
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed

def calculate_auto_thresholds(gray):
    # ヒストグラムの計算
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # 累積ヒストグラム
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()

    # 閾値の初期値を計算
    lower_threshold = np.searchsorted(cdf_normalized, 0.05) 
    upper_threshold = np.searchsorted(cdf_normalized, 0.95)

    return lower_threshold, upper_threshold

    lower_threshold = 35
    upper_threshold = 135

def main():
    # Webカメラを起動
    cap = cv2.VideoCapture(0)  # デフォルトのカメラを使用

    if not cap.isOpened():
        print("カメラが開けませんでした")
        return

    # ウィンドウの作成
    cv2.namedWindow("Edges")

    # フレームを1回取得
    ret, frame = cap.read()
    if not ret:
        print("フレームが取得できませんでした")
        cap.release()
        return

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 閾値の自動初期値を計算
    lower_threshold, upper_threshold = calculate_auto_thresholds(gray)

    lower_threshold = 35
    upper_threshold = 135


    # トラックバーの作成
    cv2.createTrackbar("Lower Threshold", "Edges", lower_threshold, 255, on_trackbar)
    cv2.createTrackbar("Upper Threshold", "Edges", upper_threshold, 255, on_trackbar)

    # カーネルの作成
    kernel_size = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    while True:
        # フレームを取得
        ret, frame = cap.read()

        if not ret:
            print("フレームが取得できませんでした")
            break

        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # モルフォロジー演算の適用（オープニング処理とクロージング処理）
        morphed_gray = apply_morphological_operations(gray, kernel)

        # トラックバーの値を取得
        lower_threshold = cv2.getTrackbarPos("Lower Threshold", "Edges")
        upper_threshold = cv2.getTrackbarPos("Upper Threshold", "Edges")

        # Cannyエッジ検出
        edges = cv2.Canny(morphed_gray, lower_threshold, upper_threshold)

        # エッジ画像と原画像の表示
        cv2.imshow("Edges", edges)
        cv2.imshow("Original", frame)

        # 'q'キーが押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # カメラを解放してウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()