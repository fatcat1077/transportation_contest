# coding: utf-8
"""
predict_speed.py

用途：透過大車與小車數量預測 5 分鐘後的整體車速，並支援：
  1. 增量式訓練（自動合併舊有資料並重訓）
  2. 多種回歸架構選擇（不含 Random Forest）
  3. 單筆預測 (--predict)
  4. 批次預測 (--input-csv / --output-csv)

回歸架構：
  linear : sklearn.linear_model.LinearRegression
  gbr    : sklearn.ensemble.GradientBoostingRegressor
  svr    : sklearn.svm.SVR
  knn    : sklearn.neighbors.KNeighborsRegressor

用法範例：
  # 增量訓練，使用 GBR
  python predict_speed.py `
    --counts C:\yolo_projects\yolov5\regression_model\carnumber\0628.csv --speed C:\yolo_projects\yolov5\regression_model\car_speed\20250628_0700-0900.csv `
    --train --regressor gbr

  # 單筆預測
  python predict_speed.py --model speed_model.pkl `
    --predict 12 34

  # 批次預測
  python predict_speed.py --model speed_model.pkl `
    --input-csv C:\yolo_projects\yolov5\regression_model\carnumber\0703.csv --output-csv pred.csv
"""

import os
import pandas as pd
import argparse
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# 合併後的歷史訓練資料檔
TRAIN_DATA_PATH = 'training_data.csv'

def load_data(counts_path, speed_path):
    """
    讀取 counts 與 speed 原始 CSV，合併並產生 target（5 分鐘後速度）
    回傳含 interval/large/small/target 的 DataFrame
    """
    counts = pd.read_csv(counts_path, parse_dates=['interval'])
    speed = pd.read_csv(speed_path)
    speed['interval'] = pd.to_datetime(speed['date'] + ' ' + speed['time_5min'])
    df = pd.merge(counts, speed[['interval', '整體車速']], on='interval', how='inner')
    df['target'] = df['整體車速'].shift(-1)
    df = df.dropna(subset=['target'])
    return df[['interval', 'large', 'small', 'target']]

def build_regressor(name):
    """
    根據名稱回傳對應的回歸模型物件
    """
    if name == 'linear':
        return LinearRegression()
    elif name == 'gbr':
        return GradientBoostingRegressor()
    elif name == 'svr':
        return SVR()
    elif name == 'knn':
        return KNeighborsRegressor(n_neighbors=5)
    else:
        raise ValueError(f"Unknown regressor: {name}")

def train_model(new_df, model_path, train_data_path, reg_name):
    """
    合併 new_df 與歷史 training_data.csv，重訓並存檔
    """
    # 讀舊資料並合併
    if os.path.exists(train_data_path):
        old_df = pd.read_csv(train_data_path, parse_dates=['interval'])
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['interval']).sort_values('interval')
    else:
        combined = new_df.copy()
    combined.to_csv(train_data_path, index=False)

    # 訓練
    X = combined[['large', 'small']]
    y = combined['target']
    model = build_regressor(reg_name)
    model.fit(X, y)
    joblib.dump(model, model_path)

    print(f"[TRAINED] {reg_name} on {len(combined)} samples → saved to {model_path}")
    print(f"[DATA] Combined history saved to {train_data_path}")

def predict_speed(model_path, large, small):
    """
    載入模型並對單筆 large/small 做預測
    """
    model = joblib.load(model_path)
    return model.predict([[large, small]])[0]

def main():
    p = argparse.ArgumentParser(
        description="5-min ahead speed prediction with various regressors"
    )
    p.add_argument('--counts', help='Path to counts CSV')
    p.add_argument('--speed', help='Path to speed CSV')
    p.add_argument('--model', default='speed_model.pkl',
                   help='Model file path')
    p.add_argument('--train', action='store_true',
                   help='Incremental train')
    p.add_argument('--regressor', choices=['linear','gbr','svr','knn'],
                   default='linear', help='Which regressor to use')
    p.add_argument('--predict', nargs=2, type=float, metavar=('LARGE','SMALL'),
                   help='Single prediction')
    p.add_argument('--input-csv',
                   help='CSV for batch prediction')
    p.add_argument('--output-csv',
                   help='Save batch predictions to CSV')
    args = p.parse_args()

    # 增量訓練
    if args.train:
        if not (args.counts and args.speed):
            print("Error: training mode requires --counts and --speed")
            return
        df_new = load_data(args.counts, args.speed)
        train_model(df_new, args.model, TRAIN_DATA_PATH, args.regressor)
        return

    # 單筆預測
    if args.predict:
        large, small = args.predict
        pred = predict_speed(args.model, large, small)
        print(f"Predicted speed: {pred:.2f}")
        return

    # 批次預測
    if args.input_csv and args.output_csv:
        df = pd.read_csv(args.input_csv, parse_dates=['interval'])
        df['predicted_speed'] = df.apply(
            lambda r: predict_speed(args.model, r['large'], r['small']), axis=1
        )
        df.to_csv(args.output_csv, index=False)
        print(f"Batch predictions saved to {args.output_csv}")
        return

    p.print_help()

if __name__ == '__main__':
    main()
