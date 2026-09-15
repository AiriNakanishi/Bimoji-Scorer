import cv2
import numpy as np

def analyze_vertical_text(image_path):
    # 1. 画像の読み込みと前処理
    img = cv2.imread(image_path)
    if img is None:
        print("画像を読み込めませんでした。パスを確認してください。")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # 白背景・黒文字を想定し、反転二値化（文字部分を255(白)にする）
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 81, 15)

    # 縦方向のパーツ（偏や旁など）を結合するための膨張処理
    # 縦方向(y)に少し長めのカーネルを設定
    # kernel = np.ones((15, 50), np.uint8) 
    kernel = np.ones((30, 80), np.uint8) 
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # 2. 輪郭抽出と外接矩形の取得
    # ここでは「スタンプ後」の画像から、文字ごとの大まかなグループを見つける
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        # これはスタンプによって太った「デカい」矩形の座標
        x, y, w, h = cv2.boundingRect(cnt)
        
        # ノイズ（小さすぎる点など）を除外
        if w > 100 and h > 100:
            # スタンプ前の「元の細い文字（thresh）」から、このデカい矩形の範囲だけを切り出す
            roi = thresh[y:y+h, x:x+w]
            
            # その切り出した範囲内にある、本当の文字のピクセル（白=255）の座標を全て取得
            points = cv2.findNonZero(roi)
            
            if points is not None:
                # 本当の文字ピクセル群にピッタリ合う「ジャストサイズ」の矩形を計算
                rx, ry, rw, rh = cv2.boundingRect(points)
                
                # 切り出した範囲でのローカル座標(rx, ry)を、画像全体のグローバル座標に直す
                final_x = x + rx
                final_y = y + ry
                
                # リストには、このジャストサイズの座標と幅・高さを追加する
                rects.append((final_x, final_y, rw, rh))

    # 縦書きなので、上から下へ（Y座標の昇順）ソート
    rects = sorted(rects, key=lambda r: r[1])

    # 3. 論文指標の計算用リスト
    aspect_ratios = []
    center_x_list = []
    gaps = []

    for i, (x, y, w, h) in enumerate(rects):
        # 縦横比 (高さ / 幅)
        aspect_ratio = h / w
        aspect_ratios.append(aspect_ratio)
        
        # 中心座標
        cx = x + w // 2
        cy = y + h // 2
        center_x_list.append(cx)
        
        # 画像への描画 (矩形: 赤, 中心点: 青)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)
        
        # 文字間の計算 (次の文字がある場合)
        if i < len(rects) - 1:
            next_x, next_y, next_w, next_h = rects[i+1]
            gap = next_y - (y + h)
            gaps.append(gap)

    # 中心線の描画（最初の文字と最後の文字の中心を結ぶ緑線）
    if len(rects) > 1:
        first_cx = center_x_list[0]
        first_cy = rects[0][1] + rects[0][3] // 2
        last_cx = center_x_list[-1]
        last_cy = rects[-1][1] + rects[-1][3] // 2
        cv2.line(img, (first_cx, first_cy), (last_cx, last_cy), (0, 255, 0), 2)

    # 4. 評価スコア（分散）の算出とコンソール出力
    print("=== 文字の分析 ===")
    for i, ar in enumerate(aspect_ratios):
        print(f"{i+1}文字目 縦横比 (H/W): {ar:.2f}")

    if len(aspect_ratios) > 0:
        ar_variance = np.var(aspect_ratios)
        print(f"-> [評価] 縦横比の分散: {ar_variance:.4f} (小さいほど形が揃っている)")

    print("\n=== 中心配置の分析 ===")
    if len(center_x_list) > 0:
        cx_variance = np.var(center_x_list)
        print(f"-> [評価] 中心X座標の分散: {cx_variance:.2f} (小さいほど一直線)")

    print("\n=== 文字間隔の分析 ===")
    for i, gap in enumerate(gaps):
        print(f"{i+1}〜{i+2}文字目の間隔: {gap} px")

    if len(gaps) > 0:
        gap_variance = np.var(gaps)
        print(f"-> [評価] 文字間隔の分散: {gap_variance:.2f} (小さいほど等間隔)")

    # 結果画像の保存
    output_path = "result_preview.png"
    cv2.imwrite(output_path, img)
    print(f"\nプレビュー画像を '{output_path}' に保存しました。")

# 実行
if __name__ == "__main__":
    # ここに用意したテスト画像のパスを指定してください
    analyze_vertical_text("analog.png")
    # analyze_vertical_text("degital.jpeg")
    # analyze_vertical_text("degital.jpeg")