import os
import random
import shutil

# 設定路徑
BASE_DIR = "all_labeled_data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TRAIN_lbl_DIR = os.path.join(BASE_DIR, "train_labels")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TEST_lbl_DIR = os.path.join(BASE_DIR, "test_labels")

# 切分比例（例如 80% 訓練，20% 測試）
TRAIN_RATIO = 0.8

# 如果目的資料夾不存在就建立
for d in [TRAIN_IMG_DIR, TRAIN_lbl_DIR, TEST_IMG_DIR, TEST_lbl_DIR]:
    os.makedirs(d, exist_ok=True)

# 取得所有影像檔案（假設副檔名為 .jpg, .png 或 .jpeg）
img_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# 隨機打亂
random.shuffle(img_files)

# 計算分界點
split_idx = int(len(img_files) * TRAIN_RATIO)
train_imgs = img_files[:split_idx]
test_imgs = img_files[split_idx:]


# 將檔案複製到對應資料夾
def copy_pair(filenames, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    for img_name in filenames:
        label_name = os.path.splitext(img_name)[0] + ".txt"  # 假設標註檔為同名 .txt
        src_img_path = os.path.join(src_img_dir, img_name)
        src_lbl_path = os.path.join(src_lbl_dir, label_name)
        dst_img_path = os.path.join(dst_img_dir, img_name)
        dst_lbl_path = os.path.join(dst_lbl_dir, label_name)
        # 複製圖片
        shutil.copy2(src_img_path, dst_img_path)
        # 如果標註檔存在才複製
        if os.path.exists(src_lbl_path):
            shutil.copy2(src_lbl_path, dst_lbl_path)
        else:
            print(f"Warning: 標註檔不存在 {src_lbl_path}")


# 執行複製
copy_pair(train_imgs, IMAGE_DIR, LABEL_DIR, TRAIN_IMG_DIR, TRAIN_lbl_DIR)
copy_pair(test_imgs, IMAGE_DIR, LABEL_DIR, TEST_IMG_DIR, TEST_lbl_DIR)

print(f"總共影像檔: {len(img_files)} 張")
print(f"訓練集: {len(train_imgs)} 張，測試集: {len(test_imgs)} 張")
