"""Two baselines for the grade-prediction task, scored on the same
train/val/test split and the same metrics as KilterCNN.

naive just predicts the training median every time, so it's a floor: if the
CNN can't beat it, the model isn't learning anything from the holds. xgboost
gets hand-engineered features (hold counts, spread, angle) instead of the
raw grid, so it's a much stronger baseline, closer to what a human might
reason about without actually seeing the spatial layout the CNN does.

needs xgboost installed (pip install xgboost). it's not in requirements.txt
since serve.py never touches it.
"""

import numpy as np
from scipy.spatial.distance import pdist
from xgboost import XGBRegressor

from dataset import KilterDataset
from model import NUM_CLASSES

MAX_ANGLE = 70.0
FEATURE_NAMES = [
    "angle_deg",
    "total_holds",
    "start_count",
    "middle_count",
    "finish_count",
    "foot_count",
    "row_span",
    "col_span",
    "diameter",
    "centroid_row",
    "row_std",
    "col_std",
    "vertical_span",
]


def build_features(images, angles):
    """turn each climb's (4, 38, 47) one-hot grid into a row of hand-picked
    numbers a tree model can split on, instead of the raw spatial grid.
    """
    n = len(images)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

    for i in range(n):
        img = images[i]
        role_counts = img.sum(axis=(1, 2))  # start, middle, finish, foot
        total = role_counts.sum()

        occ = img.sum(axis=0) > 0
        rows, cols = np.nonzero(occ)

        if total == 0:
            # guard against a holdless row rather than crashing on it
            row_span = col_span = diameter = centroid_row = row_std = col_std = 0.0
            vertical_span = 0.0
        else:
            row_span = float(rows.max() - rows.min())
            col_span = float(cols.max() - cols.min())
            centroid_row = float(rows.mean())
            row_std = float(rows.std())
            col_std = float(cols.std())

            coords = np.stack([rows, cols], axis=1)
            diameter = float(pdist(coords).max()) if len(coords) > 1 else 0.0

            start_rows = np.nonzero(img[0].sum(axis=1))[0]
            finish_rows = np.nonzero(img[2].sum(axis=1))[0]
            start_row = start_rows.mean() if len(start_rows) else centroid_row
            finish_row = finish_rows.mean() if len(finish_rows) else centroid_row
            vertical_span = float(finish_row - start_row)

        X[i] = [
            angles[i] * MAX_ANGLE,
            total,
            *role_counts,
            row_span,
            col_span,
            diameter,
            centroid_row,
            row_std,
            col_std,
            vertical_span,
        ]

    return X


def score(preds, labels, name):
    preds_rounded = np.clip(np.round(preds), 0, NUM_CLASSES - 1)
    mae = np.abs(preds - labels).mean()
    exact = (preds_rounded == labels).mean()
    within1 = (np.abs(preds_rounded - labels) <= 1).mean()
    print(f"{name:<18} MAE {mae:5.3f}   exact {exact:5.1%}   within-1 {within1:5.1%}")
    return mae, exact, within1


def main():
    train = KilterDataset("train")
    val = KilterDataset("val")
    test = KilterDataset("test")

    print(f"features: {len(FEATURE_NAMES)}  train/val/test: {len(train)}/{len(val)}/{len(test)}")
    X_train = build_features(train.images, train.angles)
    X_val = build_features(val.images, val.angles)
    X_test = build_features(test.images, test.angles)
    y_train, y_val, y_test = train.labels, val.labels, test.labels

    print()
    print(f"{'model':<18} {'metric':<0}")

    # ignores the holds completely, just the training label distribution
    median_grade = np.median(y_train)
    naive_preds = np.full(len(y_test), median_grade)
    score(naive_preds, y_test, "naive (median)")

    # early stops on val loss, same idea as the patience counter in train.py
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=0,
        early_stopping_rounds=20,
        eval_metric="mae",
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds = xgb.predict(X_test)
    score(xgb_preds, y_test, "xgboost")

    print()
    print("xgboost feature importance:")
    importances = xgb.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda t: -t[1]):
        bar = "#" * int(imp * 60)
        print(f"  {name:<14} {imp:.3f} {bar}")

    xgb.save_model("models/kilter_xgb.json")
    print("\nsaved models/kilter_xgb.json")


if __name__ == "__main__":
    main()
