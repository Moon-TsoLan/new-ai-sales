import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


FEATURE_COLS = ["category", "sub_category", "brand", "fabric", "color", "main_color", "style"]
TARGET_COLS = ["sales", "repeat_rate", "average_rating"]


def load_data(input_csv: str) -> pd.DataFrame:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"未找到输入文件: {input_csv}")

    df = pd.read_csv(input_csv)
    if "average_rating" not in df.columns and "avarage_rating" in df.columns:
        df = df.copy()
        df["average_rating"] = df["avarage_rating"]

    required_cols = FEATURE_COLS + TARGET_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"输入数据缺少字段: {missing}")

    out = df[required_cols].copy()
    for col in FEATURE_COLS:
        out[col] = out[col].fillna("unknown").astype(str).replace("", "unknown")
    for col in TARGET_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=TARGET_COLS).reset_index(drop=True)

    if len(out) < 10:
        raise RuntimeError(f"有效样本过少: {len(out)}")
    return out


def build_pipeline(random_state: int, max_depth: int | None, min_samples_leaf: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLS)],
        remainder="drop",
    )
    tree = DecisionTreeRegressor(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    model = MultiOutputRegressor(tree)
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def calc_metrics(y_true: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, target_name in enumerate(TARGET_COLS):
        yt = y_true.iloc[:, idx].values
        yp = y_pred[:, idx]
        rows.append(
            {
                "target": target_name,
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                "r2": float(r2_score(yt, yp)),
            }
        )
    return pd.DataFrame(rows)


def run_train(
    input_csv: str,
    model_path: str,
    test_size: float,
    random_state: int,
    cv_folds: int,
    max_depth: int | None,
    min_samples_leaf: int,
):
    data = load_data(input_csv)
    X = data[FEATURE_COLS]
    y = data[TARGET_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("=== 数据划分 ===")
    print(f"总样本: {len(data)}")
    print(f"训练集: {len(X_train)} ({(1 - test_size) * 100:.0f}%)")
    print(f"测试集: {len(X_test)} ({test_size * 100:.0f}%)")

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_frames = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        fold_pipe = build_pipeline(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        fold_pipe.fit(X_tr, y_tr)
        va_pred = fold_pipe.predict(X_va)

        fold_metrics = calc_metrics(y_va, va_pred)
        fold_metrics.insert(0, "fold", fold_idx)
        cv_frames.append(fold_metrics)

    cv_df = pd.concat(cv_frames, ignore_index=True)
    cv_mean_df = cv_df.groupby("target", as_index=False)[["mae", "rmse", "r2"]].mean()

    print("\n=== 交叉验证结果(每折) ===")
    print(cv_df.to_string(index=False))
    print("\n=== 交叉验证均值 ===")
    print(cv_mean_df.to_string(index=False))

    final_pipe = build_pipeline(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    final_pipe.fit(X_train, y_train)
    test_pred = final_pipe.predict(X_test)
    test_df = calc_metrics(y_test, test_pred)

    print("\n=== 测试集结果 ===")
    print(test_df.to_string(index=False))

    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final_pipe, model_path)
    print(f"\n模型已保存: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="训练多输出决策树模型（步骤三）")
    parser.add_argument(
        "--input-csv",
        default=r"C:\Users\86155\Desktop\PythonProject\data\processed\A_final_input.csv",
    )
    parser.add_argument(
        "--model-path",
        default=r"C:\Users\86155\Desktop\PythonProject\model\decision_tree_model\best_decision_tree_model.pth",
    )
    parser.add_argument("--test-size", type=float, default=0.1, help="测试集比例，默认0.1")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--max-depth", type=int, default=12, help="树最大深度")
    parser.add_argument("--min-samples-leaf", type=int, default=5, help="叶子最小样本数")
    args = parser.parse_args()

    run_train(
        input_csv=args.input_csv,
        model_path=args.model_path,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )


if __name__ == "__main__":
    main()
