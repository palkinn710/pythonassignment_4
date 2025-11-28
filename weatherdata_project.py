import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

data_dir = Path("data")
out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

dash_img = out_dir / "dashboard.png"
clean_file = out_dir / "cleaned_data.csv"
sum_file = out_dir / "building_summary.csv"
text_file = out_dir / "summary.txt"


# load all CSV files inside /data
def load_all():
    files = list(data_dir.glob("*.csv"))
    logs = []
    dfs = []

    if len(files) == 0:
        logs.append("No CSV files found.")
        return None, logs

    for file in files:
        try:
            df = pd.read_csv(file)

            if "building" not in df.columns:
                df["building"] = file.stem

            if "timestamp" not in df.columns and "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})

            dfs.append(df)
            logs.append("Loaded " + file.name)

        except Exception as e:
            logs.append("Error loading " + file.name + ": " + str(e))

    if len(dfs) > 0:
        combined = pd.concat(dfs, ignore_index=True)
        return combined, logs

    return None, logs


# create simple sample data if no CSV exists
def make_sample():
    days = pd.date_range(end=datetime.now(), periods=100, freq="D")
    buildings = ["Admin", "Lab", "Hostel"]

    rows = []

    for b in buildings:
        base = np.random.uniform(100, 400)

        for d in days:
            # weekday = higher usage, weekend = lower usage
            weekday = d.weekday()
            if weekday < 5:
                factor = 1.2
            else:
                factor = 0.8

            noise = np.random.normal(0, 20)
            kwh = base * factor + noise

            if kwh < 0:
                kwh = 0

            rows.append({
                "timestamp": d,
                "kwh": round(kwh, 2),
                "building": b
            })

    return pd.DataFrame(rows)


# clean combined dataset
def clean(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if "kwh" not in df.columns:
        raise Exception("Dataset missing kwh column")

    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df = df.dropna(subset=["kwh"])

    if "building" not in df.columns:
        df["building"] = "unknown"

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# daily totals
def daily_totals(df):
    df2 = df.set_index("timestamp")
    grp = df2.groupby("building")["kwh"].resample("D").sum()
    return grp.reset_index()


# weekly totals
def weekly_totals(df):
    df2 = df.set_index("timestamp")
    grp = df2.groupby("building")["kwh"].resample("W").sum()
    return grp.reset_index()


# building summary (mean, min, max, total)
def building_summary(df):
    g = df.groupby("building")["kwh"]
    out = g.agg(["mean", "min", "max", "sum"]).reset_index()
    out = out.rename(columns={
        "mean": "avg",
        "min": "low",
        "max": "high",
        "sum": "total"
    })
    return out


# simple OOP structure
class Reading:
    def __init__(self, ts, val):
        self.ts = pd.to_datetime(ts)
        self.val = float(val)


class Building:
    def __init__(self, name):
        self.name = name
        self.df = pd.DataFrame(columns=["timestamp", "kwh"])

    def add_df(self, d):
        self.df = pd.concat([self.df, d], ignore_index=True)

    def total(self):
        return float(self.df["kwh"].sum())

    def peak(self):
        if len(self.df) == 0:
            return None
        idx = self.df["kwh"].idxmax()
        return self.df.loc[idx]


class Manager:
    def __init__(self):
        self.buildings = {}

    def feed(self, df):
        for name, group in df.groupby("building"):
            if name not in self.buildings:
                self.buildings[name] = Building(name)
            self.buildings[name].add_df(group[["timestamp", "kwh"]])

    def table(self):
        rows = []
        for name, b in self.buildings.items():
            p = b.peak()
            if p is not None:
                rows.append({
                    "building": name,
                    "total": b.total(),
                    "peak_kwh": p["kwh"],
                    "peak_time": p["timestamp"]
                })
        return pd.DataFrame(rows)


# dashboard plot
def plot_dashboard(df):
    daily = daily_totals(df)
    wide = daily.pivot(index="timestamp", columns="building", values="kwh").fillna(0)

    weekly = weekly_totals(df)
    weekly_avg = weekly.groupby("building")["kwh"].mean()

    peak_rows = df.sort_values("kwh", ascending=False).groupby("building").head(1)

    plt.figure(figsize=(12, 10))

    # daily line plot
    ax1 = plt.subplot(3, 1, 1)
    wide.plot(ax=ax1)
    ax1.set_title("Daily Consumption")

    # weekly bar chart
    ax2 = plt.subplot(3, 1, 2)
    weekly_avg.plot(kind="bar", ax=ax2)
    ax2.set_title("Weekly Average Usage")

    # peak scatter
    ax3 = plt.subplot(3, 1, 3)
    x = range(len(peak_rows))
    y = peak_rows["kwh"].values
    names = peak_rows["building"].values

    ax3.scatter(x, y)

    for i in range(len(names)):
        ax3.text(x[i], y[i], names[i], ha="center", va="bottom")

    ax3.set_xticks(list(x))
    ax3.set_xticklabels(names)
    ax3.set_title("Peak Usage (Per Building)")

    plt.tight_layout()
    plt.savefig(dash_img)
    plt.close()


# writing outputs
def write_outputs(df, summ):
    df.to_csv(clean_file, index=False)
    summ.to_csv(sum_file, index=False)

    total_usage = df["kwh"].sum()
    top = summ.sort_values("total", ascending=False).iloc[0]

    idx = df["kwh"].idxmax()
    peak = df.loc[idx]

    with open(text_file, "w") as f:
        f.write("Campus Energy Summary\n\n")
        f.write("Total usage: " + str(total_usage) + " kWh\n")
        f.write("Top building: " + top["building"] + " (" + str(top["total"]) + " kWh)\n")
        f.write("Peak reading: " + str(peak["kwh"]) + " kWh at " + str(peak["timestamp"]) + "\n")

    print("Saved:", clean_file)
    print("Saved:", sum_file)
    print("Saved:", text_file)
    print("Saved dashboard:", dash_img)


def main():
    df, logs = load_all()
    for l in logs:
        print(l)

    if df is None:
        df = make_sample()

    df = clean(df)

    mgr = Manager()
    mgr.feed(df)

    summ = building_summary(df)
    plot_dashboard(df)
    write_outputs(df, summ)

    print("Done.")


if __name__ == "__main__":
    main()
