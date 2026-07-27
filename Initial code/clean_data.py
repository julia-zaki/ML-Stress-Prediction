"""
clean_data.py – Clean and split the training data into train/val/test sets.

Outputs:
    train.csv, val.csv, test.csv
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

ID_COL = "id"  # person key — the split is done on this
DAY_COL = "day_in_study"  # kept as an identifier, not a feature
TARGET = "stress"  # label to predict (already High/Low/etc.)

NUM_COLS = [
    "estrogen", "rmssd", "hrv_hf", "resting_hr",
    "hr_bpm", "calories", "wrist_temp_diff", "glucose",
]
# will be one-hot encoded below
CATEGORICAL_COLS = ["phase"]


def main(input_csv):
    df = pd.read_csv(input_csv)
    print(f'Loaded {input_csv}: {df.shape[0]} rows, {df.shape[1]} columns')

    # Split by unique_id (60/15/25)
    unique_ids = df['unique_id'].unique()
    ids_trainval, ids_test = train_test_split(unique_ids, test_size=0.25, random_state=42)
    ids_train, ids_val = train_test_split(ids_trainval, test_size=0.20, random_state=42)

    splits = {
        'train': df[df['unique_id'].isin(ids_train)].copy(),
        'val':   df[df['unique_id'].isin(ids_val)].copy(),
        'test':  df[df['unique_id'].isin(ids_test)].copy(),
    }
    for name, sdf in splits.items():
        print(f'  {name}: {len(sdf)} rows  ({sdf["Painting"].value_counts().to_dict()})')

    # For now, no need to change stress to numbers

    # Cliping outliers in numerical columns (using train-set 95th percentile)
    clip_upper = {}
    for col in NUM_COLS:
        clip_upper[col] = splits['train'][col].quantile(0.95)
    for sdf in splits.values():
        for col in NUM_COLS:
            sdf[col] = sdf[col].clip(upper=clip_upper[col])

    # Imputing missing numericals with per-stress mean from train only
    train_means = {}
    for col in NUM_COLS:
        train_means[col] = splits["train"].groupby(TARGET)[col].mean()

    for sdf in splits.values():
        for col in NUM_COLS:
            for cls, mean_val in train_means[col].items():
                mask = (sdf[TARGET] == cls) & sdf[col].isna()
                sdf.loc[mask, col] = mean_val
        # When can't fill a blank because a stress class has no training data for that column.
        for col in NUM_COLS:
            sdf[col] = sdf[col].fillna(splits["train"][col].mean())

    # One-hot encode the stress
    onehot_parts = {}
    for name, sdf in splits.items():
        onehot_parts[name] = pd.get_dummies(
            sdf[CATEGORICAL_COLS], columns=CATEGORICAL_COLS
        ).astype(int)
    # align columns across splits
    all_cols = onehot_parts["train"].columns
    for name in onehot_parts:
        onehot_parts[name] = onehot_parts[name].reindex(columns=all_cols, fill_value=0)


    # Assembling feature matrix
    # now we have one complete table per split
    feature_frames = {}
    for name, sdf in splits.items():
        parts = [
            sdf[[ID_COL, DAY_COL, TARGET]].reset_index(drop=True),
            sdf[NUM_COLS].reset_index(drop=True), # re-numbering the rows
            onehot_parts[name].reset_index(drop=True),
        ]
        feature_frames[name] = pd.concat(parts, axis=1)

    # Normalize numerical features w.r.t. train set
    # Compute mean and std from train set only
    train_mean = feature_frames["train"][NUM_COLS].mean()
    train_std = feature_frames["train"][NUM_COLS].std().replace(0, 1)

    for name in feature_frames:
        feature_frames[name][NUM_COLS] = (
                (feature_frames[name][NUM_COLS] - train_mean) / train_std
        )

    # Saving
    for name, fdf in feature_frames.items():
        out_path = f'{name}.csv'
        fdf.to_csv(out_path, index=False)
        print(f'Saved {out_path}: {fdf.shape}')

    # Print final feature summary
    print(f'\nFinal features ({feature_frames["train"].shape[1]} columns):')
    print('  Columns:', list(feature_frames['train'].columns))


if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else "merged.csv"
    main(input_file)
    print("clean_data.py ran successfully")
