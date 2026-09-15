import cv2
import numpy as np
import os
import csv
import glob

# ==========================================
# 【基本設定】
# 記録する最大文字数（宛名など長い文字列を評価する際はここを増やします）
MAX_CHARS = 10 
# ==========================================


def analyze_vertical_text(image_path):
    print(f"処理中: {image_path} ...")
    
    # 1. 画像の読み込みと前処理
    img = cv2.imread(image_path)
    if img is None:
        print(f"エラー: {image_path} を読み込めませんでした。")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 適応的二値化（影や照明ムラを除去）
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 81, 15)

    # 縦方向のパーツを結合するための膨張処理
    kernel = np.ones((15, 50), np.uint8) 
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # 2. 輪郭抽出と外接矩形の取得
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # ノイズの足切り
        if w > 50 and h > 50:
            roi = thresh[y:y+h, x:x+w]
            points = cv2.findNonZero(roi)
            
            if points is not None:
                rx, ry, rw, rh = cv2.boundingRect(points)
                final_x = x + rx
                final_y = y + ry
                rects.append((final_x, final_y, rw, rh))

    # 縦書きなので、上から下へ（Y座標の昇順）ソート
    rects = sorted(rects, key=lambda r: r[1])

    # 3. 論文指標の計算用リスト
    aspect_ratios = []
    center_x_list = []
    gaps = []

    for i, (x, y, w, h) in enumerate(rects):
        aspect_ratios.append(h / w)
        
        cx = x + w // 2
        cy = y + h // 2
        center_x_list.append(cx)
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)
        
        if i < len(rects) - 1:
            next_y = rects[i+1][1]
            gap = next_y - (y + h)
            gaps.append(gap)

    if len(rects) > 1:
        first_cx = center_x_list[0]
        first_cy = rects[0][1] + rects[0][3] // 2
        last_cx = center_x_list[-1]
        last_cy = rects[-1][1] + rects[-1][3] // 2
        cv2.line(img, (first_cx, first_cy), (last_cx, last_cy), (0, 255, 0), 2)

    # 4. ファイル名の解析（IDと条件の分割）
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    parts = name_without_ext.split('_')
    if len(parts) >= 2:
        subject_id = parts[0]
        condition = parts[1]
    else:
        subject_id = name_without_ext
        condition = "unknown"

    # 5. CSVファイルへのデータ出力
    csv_path = "evaluation_results.csv"
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            header = ["被験者ID", "条件", "元ファイル名", "文字数", "縦横比_分散", "中心X_分散", "文字間隔_分散"]
            for i in range(1, MAX_CHARS + 1):
                header.append(f"{i}文字目_縦横比")
            for i in range(1, MAX_CHARS + 1):
                header.append(f"{i}文字目_中心X")
            for i in range(1, MAX_CHARS):
                header.append(f"{i}-{i+1}文字間")
            writer.writerow(header)
        
        ar_v = np.var(aspect_ratios) if len(aspect_ratios) > 0 else 0
        cx_v = np.var(center_x_list) if len(center_x_list) > 0 else 0
        gap_v = np.var(gaps) if len(gaps) > 0 else 0
        
        def get_val(lst, index, decimals=2):
            return round(lst[index], decimals) if index < len(lst) else ""

        ar_cols = [get_val(aspect_ratios, i) for i in range(MAX_CHARS)]
        cx_cols = [get_val(center_x_list, i, 0) for i in range(MAX_CHARS)]
        gap_cols = [get_val(gaps, i, 0) for i in range(MAX_CHARS - 1)]
        
        row_data = [
            subject_id, condition, filename, len(rects), 
            round(ar_v, 4), round(cx_v, 2), round(gap_v, 2)
        ] + ar_cols + cx_cols + gap_cols
        
        writer.writerow(row_data)
        
    output_path = f"result_{filename}"
    cv2.imwrite(output_path, img)


# --- ここから下が「一括処理（バッチ処理）」の仕組み ---
if __name__ == "__main__":
    target_folder = "images"
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"'{target_folder}' フォルダを作成しました。")
        print(f"このフォルダに分析したい画像を入れ、もう一度実行してください。")
    else:
        image_files = glob.glob(os.path.join(target_folder, "*.png"))
        image_files += glob.glob(os.path.join(target_folder, "*.jpg"))
        
        if len(image_files) == 0:
            print(f"'{target_folder}' フォルダ内に画像が見つかりません。")
        else:
            print(f"合計 {len(image_files)} 枚の画像を処理します...\n")
            for img_path in image_files:
                analyze_vertical_text(img_path)
            
            print("\nすべての処理が完了しました！")