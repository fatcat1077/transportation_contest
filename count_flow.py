#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
from datetime import datetime
import argparse

def parse_filename_timestamp(path):
    """
    從檔名 (e.g. 2025-07-05_07-00-04.txt) 解析 datetime 物件
    """
    fn = os.path.basename(path)
    name, _ = os.path.splitext(fn)
    # 檔名格式 YYYY-MM-DD_HH-MM-SS
    return datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")

def floor_to_5min(dt):
    """
    把 datetime dt 的分鐘向下取整到 5 的倍數，秒數歸零
    """
    minute = (dt.minute // 5) * 5
    return dt.replace(minute=minute, second=0, microsecond=0)

def process_labels(label_dir):
    """
    讀取所有 txt，做過濾、分組、計數後回傳 DataFrame
    """
    records = []
    for txt_path in glob.glob(os.path.join(label_dir, "*.txt")):
        ts = parse_filename_timestamp(txt_path)
        # 讀檔：每行 [class, x_ctr, y_ctr, w, h]
        df = pd.read_csv(txt_path, sep=r"\s+", header=None,
                         names=["class", "x", "y", "w", "h"])
        # 篩選對向車道：留下 0.74*x + y - 0.94 >= 0 的行
        df = df[(0.74 * df["x"] + df["y"] - 0.94) >= 0]
        # 向下取整到 5 分鐘
        interval = floor_to_5min(ts)
        # 統計數量（class 0→大車，class 1→小車）
        n_large = int((df["class"] == 0).sum())
        n_small = int((df["class"] == 1).sum())
        records.append({
            "interval": interval,
            "large": n_large,
            "small": n_small
        })

    if not records:
        print("No label files found in", label_dir)
        return None

    all_df = pd.DataFrame(records)
    # 依 interval 分組、加總
    result = all_df.groupby("interval", as_index=False).sum()
    # 按時間排序
    result = result.sort_values("interval")
    return result

def main():
    parser = argparse.ArgumentParser(
        description="統計每 5 分鐘的大／小車數量並輸出 CSV")
    parser.add_argument(
        "--label_dir", required=True,
        help="存放 YOLO 標注 txt 檔案的資料夾路徑")
    parser.add_argument(
        "--output", default="vehicle_counts.csv",
        help="輸出的 CSV 檔名 (預設: vehicle_counts.csv)")
    args = parser.parse_args()

    df = process_labels(args.label_dir)
    if df is None:
        return

    # 將時間格式轉成字串（方便匯出）
    df["interval"] = df["interval"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved summary to {args.output}")

if __name__ == "__main__":
    main()
