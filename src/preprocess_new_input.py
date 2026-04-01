import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class PreprocessConfig:
    input_csv: str = os.path.join("data", "raw", "new_input.csv")
    output_csv: str = os.path.join("data", "processed", "new_input_processed.csv")

    required_cols: tuple[str, ...] = ("discount", "images")
    drop_cols: tuple[str, ...] = (
        "_id",
        "crawled_at",
        "images",
        "out_of_stock",
        "product_details",
        "seller",
        "url",
    )


def _norm_text(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def preprocess(cfg: PreprocessConfig) -> dict:
    if not os.path.exists(cfg.input_csv):
        raise FileNotFoundError(f"输入文件不存在: {cfg.input_csv}")

    os.makedirs(os.path.dirname(cfg.output_csv), exist_ok=True)

    df = pd.read_csv(cfg.input_csv)
    original_rows = len(df)

    missing_required = [c for c in cfg.required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"缺少必要字段: {missing_required}")

    # 1) 去掉 discount 或 images 为空的数据
    discount_series = df["discount"].map(_norm_text)
    images_series = df["images"].map(_norm_text)
    df = df[(discount_series != "") & (images_series != "")].copy()
    after_remove_empty = len(df)

    # 3) 交换 discount 和 description 的值
    if "description" not in df.columns:
        raise ValueError("缺少字段 description，无法与 discount 交换。")
    temp = df["discount"].copy()
    df["discount"] = df["description"]
    df["description"] = temp

    # 2) 删除指定字段
    existing_drop_cols = [c for c in cfg.drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop_cols, errors="ignore")

    # 4) 保存
    df.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")

    return {
        "input_csv": cfg.input_csv,
        "output_csv": cfg.output_csv,
        "original_rows": original_rows,
        "rows_after_remove_empty_discount_images": after_remove_empty,
        "final_rows": len(df),
        "dropped_columns_applied": existing_drop_cols,
        "output_columns": list(df.columns),
    }


if __name__ == "__main__":
    config = PreprocessConfig()
    report = preprocess(config)
    print("=== new_input 预处理完成 ===")
    for k, v in report.items():
        print(f"{k}: {v}")

