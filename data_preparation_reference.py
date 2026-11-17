"""
STEP 1: Load, clean and explore air-quality data.

Goal of the whole project: predict tomorrow's PM2.5 (a harmful fine-dust
pollutant) from the last 14 days of pollution + weather readings.

Run:
    python step1_data_prep.py                 -> uses demo data (to test the code)
    python step1_data_prep.py my_lucknow.csv  -> uses your REAL data
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # lets plots be saved without a screen
import matplotlib.pyplot as plt

TARGET = "PM2.5"
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]


def make_demo_data(n_days=1500, seed=0):
    """FAKE data only to test that the pipeline runs. NEVER report results from this."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    season = 1 + 0.6 * np.cos(2 * np.pi * (dates.dayofyear - 20) / 365)  # winter peak
    pm25 = np.zeros(n_days)
    pm25[0] = 100
    for t in range(1, n_days):  # today depends on yesterday (autocorrelation)
        pm25[t] = 0.7 * pm25[t - 1] + 0.3 * 90 * season[t] + rng.normal(0, 12)
    pm25 = np.clip(pm25, 5, None)
    df = pd.DataFrame({
        "Date": dates,
        "PM2.5": pm25,
        "PM10": pm25 * 1.8 + rng.normal(0, 15, n_days),
        "NO2": 30 + 0.2 * pm25 + rng.normal(0, 6, n_days),
        "SO2": 10 + rng.normal(0, 2, n_days),
        "CO": 1 + 0.01 * pm25 + rng.normal(0, 0.2, n_days),
        "O3": 40 - 0.1 * pm25 + rng.normal(0, 8, n_days),
    })
    # punch random holes, like real sensors that go offline
    for col in FEATURES:
        df.loc[rng.choice(n_days, 60, replace=False), col] = np.nan
    return df


def load_data(path=None):
    if path is None:
        print("!! Using DEMO data. Replace with real CSV before reporting results.")
        return make_demo_data()
    df = pd.read_csv(path)
    # Real CPCB files often name the date column 'Date', 'date' or 'From Date'.
    date_col = next(c for c in df.columns if "date" in c.lower())
    df = df.rename(columns={date_col: "Date"})
    iso = df["Date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.5
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=not iso)
    return df


def clean(df):
    df = df[["Date"] + [c for c in FEATURES if c in df.columns]].copy()
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.groupby("Date").mean(numeric_only=True)          # one row per day
    df = df.asfreq("D")                                       # insert missing days
    print("Missing values per column BEFORE filling:\n", df.isna().sum(), "\n")
    df = df.interpolate(limit=7, limit_direction="both")      # fill short gaps only
    df = df.dropna()                                          # drop long gaps
    df = df[df[TARGET] > 0]                                   # impossible values
    return df


def explore(df, out_dir="."):
    print(df.describe().round(1), "\n")
    print("Correlation of each pollutant with PM2.5:")
    print(df.corr()[TARGET].round(2), "\n")

    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    df[TARGET].plot(ax=ax[0], lw=0.7)
    ax[0].set_title("Daily PM2.5 over time")
    ax[0].set_ylabel("ug/m3")
    df.groupby(df.index.month)[TARGET].mean().plot.bar(ax=ax[1])
    ax[1].set_title("Average PM2.5 by month (seasonality)")
    ax[1].set_xlabel("Month")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_plots.png", dpi=120)
    print("Saved eda_plots.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data = clean(load_data(path))
    explore(data)
    data.to_csv("clean_data.csv")
    print(f"\nSaved clean_data.csv with {len(data)} days "
          f"({data.index.min().date()} to {data.index.max().date()})")
