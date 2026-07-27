"""
merge_data.py – Merge selected columns from raw CSVs into one table.
"""
import pandas as pd

KEY = ["id", "day_in_study"]
FILES = {
    "cycle":  ("hormones_and_selfreport.csv", ["phase", "estrogen", "stress"]),
    "hrv":    ("hrv.csv",         ["rmssd", "high_frequency"]),
    "resting_hr":     ("resting_heart_rate.csv",  {"value": "resting_hr"}),
    "hr":     ("heart_rate.csv",  ["bpm"]),
    "calories": ("calories.csv",      ["calories"]),
    "wrist_temperature": ("wrist_temperature.csv",      ["temperature_diff_from_baseline"]),
    "glucose": ("glucose.csv",      ["glucose_value"]),
}


def main():
    merged = None

    for label, (path, col_map) in FILES.items():
        df = pd.read_csv(path)
        required_columns = KEY + list(col_map.keys())
        df = df[required_columns].rename(columns=col_map)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=KEY, how="outer")  # for now missing data is NaN

    # percentage missing values
    n_rows = len(merged)
    counts = merged.isna().sum()
    pct = (counts / n_rows * 100).round(1)
    have_gaps = counts[counts > 0]
    if len(have_gaps) == 0:
        print("\nNo missing values in any column.")
    else:
        print(f"\nMissing values (of {n_rows} rows):")
        for col in have_gaps.index:
            print(f"  {col:<22} {counts[col]:>6}  ({pct[col]}%)")


if __name__ == "__main__":
    main()
