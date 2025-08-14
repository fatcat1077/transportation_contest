from datetime import datetime, date, timedelta
import pandas as pd
from speed_app import download_for_date, summarize_day, VEHICLES

# 批次日期範圍
start = datetime.strptime("20250701", "%Y%m%d").date()
end   = datetime.strptime("20250706", "%Y%m%d").date()

# 時段設定：一天抓 0~23 時所有小時
hours = [f"{h:02d}" for h in range(0, 24)]

for day in pd.date_range(start, end):
    day = day.date()
    day_str = day.strftime("%Y%m%d")
    print(f"處理 {day_str} …")

    # 1) 下載該日資料
    download_for_date(day, hours)

    # 2) 統計該日 車速／流量
    wide = summarize_day(day_str, hours)
    if wide is None:
        print(f"  → {day_str} 無資料，跳過")
        continue

    # 3) 抽出「車速」欄位，並存成 CSV
    speed_cols = [f"{veh}_車速" for veh in VEHICLES.values() if f"{veh}_車速" in wide.columns]
    out = wide[["date", "time_5min"] + speed_cols]
    out.to_csv(f"{day_str}_speeds.csv", index=False, encoding="utf-8-sig")
    print(f"  → 存檔: {day_str}_speeds.csv")
