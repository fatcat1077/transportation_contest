import threading
import tkinter as tk
from datetime import date, datetime, time, timedelta
from pathlib import Path
from tkinter import messagebox
from tkinter.ttk import Combobox

import matplotlib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt

# ==== 全域參數設定 ====
BASE_URL = "https://tisvcloud.freeway.gov.tw/history/TDCS/M05A/"
DATA_ROOT = Path(r"C:\Coding\Python\carSpeed")
SUMMARY_ROOT = Path(r"C:\Coding\Python\每日速度")
CHART_ROOT = Path(r"C:\Coding\Python\每日速度圖表")

S1, S2 = "01F0928S", "01F0950S"
VEHICLES = {"31": "小客車", "32": "小貨車", "41": "大客車", "42": "大貨車", "5": "聯結車"}

# PCU 因子
PCU = {"小客車": 1.0, "小貨車": 1.5, "大客車": 2.0, "大貨車": 2.5, "聯結車": 3.0}

# LOS 分界
LOS_BREAKS = [0, 9, 18, 27, 36, 45, float("inf")]
LOS_LABELS = ["A", "B", "C", "D", "E", "F"]

matplotlib.rc("font", family="Microsoft Yahei")


def make_time_slots():
    slots = []
    start = datetime.combine(date.today(), time(0, 0))
    end = datetime.combine(date.today(), time(23, 30))
    t = start
    while t <= end:
        slots.append(t.time().strftime("%H:%M"))
        t += timedelta(minutes=30)
    return slots


TIME_SLOTS = make_time_slots()


def download_for_date(day: date, hours: list[str]):
    day_str = day.strftime("%Y%m%d")
    for hr in hours:
        url = f"{BASE_URL}{day_str}/{hr}/"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
        except:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        links = [a["href"] for a in soup.select("a[href]") if a["href"] != "../"]
        outdir = DATA_ROOT / day_str / hr
        outdir.mkdir(parents=True, exist_ok=True)
        for fn in links:
            tgt = outdir / fn
            if tgt.exists():
                continue
            try:
                rr = requests.get(url + fn, timeout=20)
                rr.raise_for_status()
                tgt.write_bytes(rr.content)
            except:
                pass


def summarize_day(day_str: str, hours: list[str]) -> pd.DataFrame | None:
    recs = []
    base = DATA_ROOT / day_str
    for hr in hours:
        fld = base / hr
        if not fld.is_dir():
            continue
        for fp in fld.glob("TDCS_M05A_*.csv"):
            ts = datetime.strptime(fp.stem.split("_")[-2] + fp.stem.split("_")[-1], "%Y%m%d%H%M%S")
            df = pd.read_csv(
                fp,
                header=None,
                skiprows=1,
                usecols=[1, 2, 3, 4, 5],
                names=["s1", "s2", "veh", "speed", "vol"],
                dtype=str,
            )
            df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
            df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
            df = df[(df["s1"].str.strip() == S1) & (df["s2"].str.strip() == S2)]
            if df.empty:
                continue
            df = df.assign(date=ts.date(), time_5min=ts.time())
            recs.append(df[["date", "time_5min", "veh", "speed", "vol"]])
    if not recs:
        return None
    long = pd.concat(recs, ignore_index=True)
    long["veh"] = long["veh"].map(VEHICLES).fillna(long["veh"])
    wide = long.pivot_table(
        index=["date", "time_5min"], columns="veh", values=["speed", "vol"], aggfunc={"speed": "mean", "vol": "sum"}
    )
    wide.columns = [f"{veh}_{'車速' if m == 'speed' else '流量'}" for m, veh in wide.columns]
    wide = wide.reset_index()
    cols = ["date", "time_5min"]
    for v in VEHICLES.values():
        cols += [f"{v}_車速", f"{v}_流量"]
    return wide.reindex(columns=cols + [c for c in wide.columns if c not in cols])


def export_and_plot(wide: pd.DataFrame, day_str: str, start_t: time, end_t: time, is_today: bool, root: tk.Tk):
    # 0) 計算各車種密度 & LOS
    for veh in VEHICLES.values():
        vol = wide[f"{veh}_流量"] * 12
        wide[f"{veh}_密度"] = vol / wide[f"{veh}_車速"]
        wide[f"{veh}_LOS"] = pd.cut(wide[f"{veh}_密度"], bins=LOS_BREAKS, labels=LOS_LABELS, right=False)
    # PCU 加權整體
    pcu5, spdpcu = [], []
    for veh, pcu in PCU.items():
        if f"{veh}_流量" in wide:
            wide[f"{veh}_PCU5"] = wide[f"{veh}_流量"] * pcu
            pcu5.append(f"{veh}_PCU5")
            wide[f"{veh}_SPD×PCU"] = wide[f"{veh}_車速"] * wide[f"{veh}_PCU5"]
            spdpcu.append(f"{veh}_SPD×PCU")
    wide["PCU_q"] = wide[pcu5].sum(axis=1) * 12
    wide["v_weighted"] = wide[spdpcu].sum(axis=1) / wide[pcu5].sum(axis=1)
    wide["k_total"] = wide["PCU_q"] / wide["v_weighted"]
    wide["LOS_total"] = pd.cut(wide["k_total"], bins=LOS_BREAKS, labels=LOS_LABELS, right=False)

    # 1) 匯出 Excel，檔名加入時段
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    range_tag = f"{start_t.strftime('%H%M')}-{end_t.strftime('%H%M')}"
    excel_path = SUMMARY_ROOT / f"{day_str}_{range_tag}.xlsx"
    wide.to_excel(excel_path, index=False)

    # 2) 篩時段
    if is_today:
        now = datetime.now().time()
        end_t = min(end_t, now)
    sub = wide[wide["time_5min"].between(start_t, end_t)]

    # 3) 單車種圖
    CHART_ROOT.mkdir(parents=True, exist_ok=True)
    times = sub["time_5min"].astype(str).tolist()
    idxs = list(range(len(times)))
    for veh in VEHICLES.values():
        if f"{veh}_流量" not in sub:
            continue
        yb = sub[f"{veh}_流量"].fillna(0).tolist()
        yl = sub[f"{veh}_車速"].fillna(0).tolist()
        los_t = sub["LOS_total"].astype(str).tolist()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(idxs, yb, color="gray")
        ax.set_xticks(idxs)
        ax.set_xticklabels(times, rotation=45, ha="right")
        ax.set_ylabel("流量(輛/5分鐘)")
        for x, v in zip(idxs, yb):
            ax.text(x, v / 2, f"{int(v)}", ha="center", va="center", color="white", fontsize=8)

        ax2 = ax.twinx()
        ax2.plot(idxs, yl, color="firebrick", marker="o")
        ax2.set_ylabel("速度(km/h)")
        ax2.set_ylim(40, 140)
        for x, v in zip(idxs, yl):
            ax2.text(x, v + 1, f"{v:.1f}", ha="center", va="bottom", color="firebrick", fontsize=8)
        for x, l in zip(idxs, los_t):
            base = yl[x] if x < len(yl) else 0
            ax2.text(x, base + 5, l, ha="center", va="bottom", color="black", fontsize=7)

        ax2.legend(loc="upper left")
        plt.title(f"{day_str} {veh} {range_tag} 圖表")
        fig.subplots_adjust(left=0.07, right=0.93, top=0.85, bottom=0.2)
        fig.savefig(CHART_ROOT / f"{day_str}_{veh}_{range_tag}.png")
        plt.close(fig)

    # 4) 整體圖：總流量／流率＋LOS_total
    sub["total_vol5"] = sub[[f"{v}_流量" for v in VEHICLES.values()]].sum(axis=1)
    sub["total_flow_rate"] = sub["total_vol5"] * 12
    los_all = sub["LOS_total"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(idxs, sub["total_vol5"], color="skyblue", label="總流量 (5 分鐘)")
    ax.set_xticks(idxs)
    ax.set_xticklabels(times, rotation=45, ha="right")
    ax.set_ylabel("5 分鐘總流量(輛)")

    ax2 = ax.twinx()
    ax2.plot(idxs, sub["total_flow_rate"], color="navy", marker="s", label="流率 (veh/h)")
    ax2.set_ylabel("流率 (veh/h)")
    ax2.set_ylim(0, sub["total_flow_rate"].max() * 1.2)
    for x, y, l in zip(idxs, sub["total_flow_rate"], los_all):
        ax2.text(x, y + sub["total_flow_rate"].max() * 0.03, l, ha="center", va="bottom", color="black", fontsize=8)

    lines, labels = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lab2, loc="upper left")
    plt.title(f"{day_str} 整體 {range_tag} 組合圖")
    fig.subplots_adjust(left=0.07, right=0.93, top=0.85, bottom=0.2)
    fig.savefig(CHART_ROOT / f"{day_str}_整體_{range_tag}.png")
    plt.close(fig)

    # 5) 提示完成
    overall = sub["LOS_total"].mode().iloc[0] if not sub.empty else ""
    root.after(0, lambda: messagebox.showinfo("完成", f"{day_str} 圖表已生成\n範圍：{range_tag}\n整體LOS：{overall}"))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("高速公路即時圖表產生器")
        self.geometry("400x380")

        f = tk.Frame(self)
        f.pack(pady=10)
        tk.Label(f, text="開始：").grid(row=0, column=0, padx=5)
        self.cb1 = Combobox(f, values=TIME_SLOTS, width=6, state="readonly")
        self.cb1.set("07:00")
        self.cb1.grid(row=0, column=1)
        tk.Label(f, text="結束：").grid(row=0, column=2, padx=5)
        self.cb2 = Combobox(f, values=TIME_SLOTS, width=6, state="readonly")
        self.cb2.set("09:00")
        self.cb2.grid(row=0, column=3)

        tk.Button(self, text="產生即時圖表", width=25, command=self.run_today).pack(pady=8)

        f2 = tk.Frame(self)
        f2.pack(pady=8)
        tk.Label(f2, text="日期(YYYYMMDD)：").pack(side="left")
        self.date_entry = tk.Entry(f2, width=10)
        self.date_entry.insert(0, date.today().strftime("%Y%m%d"))
        self.date_entry.pack(side="left")
        tk.Button(self, text="產生指定日期圖表", width=25, command=self.run_date).pack(pady=8)

    def run_today(self):
        def job():
            now = datetime.now()
            day_dt = date.today()
            ds = day_dt.strftime("%Y%m%d")
            start_hour = max(7, now.hour - 2)
            hours = [f"{h:02d}" for h in range(start_hour, now.hour + 1)]
            download_for_date(day_dt, hours)
            wide = summarize_day(ds, hours)
            if wide is None:
                return self.after(0, lambda: messagebox.showerror("錯誤", "無資料"))
            st = (now - timedelta(hours=2)).time()
            et = now.time()
            export_and_plot(wide, ds, st, et, True, self)

        threading.Thread(target=job, daemon=True).start()

    def run_date(self):
        def job():
            s = self.date_entry.get()
            try:
                day_dt = datetime.strptime(s, "%Y%m%d").date()
            except:
                return self.after(0, lambda: messagebox.showerror("錯誤", "日期格式錯誤"))
            ds = day_dt.strftime("%Y%m%d")
            st_obj = datetime.strptime(self.cb1.get(), "%H:%M")
            et_obj = datetime.strptime(self.cb2.get(), "%H:%M")
            hours = [f"{h:02d}" for h in range(st_obj.hour, et_obj.hour + 1)]
            download_for_date(day_dt, hours)
            wide = summarize_day(ds, hours)
            if wide is None:
                return self.after(0, lambda: messagebox.showerror("錯誤", "無資料"))
            st = st_obj.time()
            et = et_obj.time()
            export_and_plot(wide, ds, st, et, False, self)

        threading.Thread(target=job, daemon=True).start()


if __name__ == "__main__":
    print(">>> 程式啟動，準備開啟 GUI")
    App().mainloop()
    print(">>> mainloop() 結束")
