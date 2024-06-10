import cv2
import numpy as np

drawing = False  # マウスで描画中かどうか
brush_size = 50  # ブラシのサイズ
selected_area = None  # ブラシで選択された領域

#事前に決める箇所
luminance_value = 40  #境界の輝度値の範囲
image_path = "img1.jpg" #画像のパス

# マウスコールバック関数
def paint(event, x, y, flags, param):
    global drawing, selected_area

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(selected_area, (x, y), brush_size, (255, 255, 255), -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(selected_area, (x, y), brush_size, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(selected_area, (x, y), brush_size, (255, 255, 255), -1)

# 画像を読み込む部分
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"画像の読み込みに失敗しました。パスが正しいか確認してください: {image_path}")

clone = image.copy()
selected_area = np.zeros(image.shape[:2], dtype=np.uint8)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", paint)

# 画像を表示し、描画するのを待つ。影の部分をマークする。cキーを押せば完了。
while True:
    display_image = image.copy()
    display_image[selected_area == 255] = [0, 255, 0]  # ブラシの色は緑
    cv2.imshow("image", display_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        break

# 選択された部分（影の部分）の輝度値を計算
selected_mask = selected_area == 255
selected_hsv = cv2.cvtColor(clone, cv2.COLOR_BGR2HSV)
selected_v = selected_hsv[:, :, 2][selected_mask]
mean_selected_v = np.mean(selected_v)

print(f"選択された部分（緑マーカー）の平均輝度値: {mean_selected_v}")

# 画像全体で輝度値を計算し、選択部分の輝度値と比較
image_hsv = cv2.cvtColor(clone, cv2.COLOR_BGR2HSV)
image_v = image_hsv[:, :, 2]

# 選択された部分よりも明るい部分で、輝度差が40以上の部分をマーク
diff_mask = (image_v - mean_selected_v) >= luminance_value

# 影でない部分を赤でマーク
red_marked_image = clone.copy()
red_marked_image[diff_mask] = [0, 0, 255]

# 影の部分を白でマーク
white_marked_image = np.zeros_like(clone)
white_marked_image[~diff_mask] = [255, 255, 255]

# 赤マークの輝度値を計算
red_v = image_v[diff_mask]
mean_red_v = np.mean(red_v)

# 白マークの輝度値を計算
white_v = image_v[~diff_mask]
mean_white_v = np.mean(white_v)

print(f"赤マーク部分の平均輝度値: {mean_red_v}")
print(f"白マーク部分の平均輝度値: {mean_white_v}")

#ここから影を消すフェーズ

# 輝度差を計算
brightness_difference = mean_red_v - mean_white_v

# 影の部分の輝度値を調整。影と光の輝度差を影の輝度値に加える。
shadow_mask = ~diff_mask
shadow_hsv = image_hsv.copy()
shadow_hsv[:, :, 2][shadow_mask] = np.clip(shadow_hsv[:, :, 2][shadow_mask] + brightness_difference, 0, 255)
adjusted_image = cv2.cvtColor(shadow_hsv, cv2.COLOR_HSV2BGR)

# 最後に選択した部分（緑マーカー）の輝度値を再度表示
final_selected_hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
final_selected_v = final_selected_hsv[:, :, 2][selected_mask]
final_mean_selected_v = np.mean(final_selected_v)

print(f"最終的な選択部分（緑マーカー）の平均輝度値: {final_mean_selected_v}")

# 結果を表示して保存
cv2.imshow("Red Marked Image", red_marked_image)
cv2.imshow("White Marked Image", white_marked_image)
cv2.imshow("Adjusted Image", adjusted_image)
cv2.imwrite("red_marked_image.jpg", red_marked_image)
cv2.imwrite("white_marked_image.jpg", white_marked_image)
cv2.imwrite("adjusted_image.jpg", adjusted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()