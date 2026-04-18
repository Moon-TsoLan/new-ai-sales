import argparse
import os
from dataclasses import dataclass

from image_style_module import build_style_output, collect_image_style_data, load_style_model

# 利用现有的Kmeans模型提取风格和主颜色
@dataclass
class ExtractConfig:
    input_csv: str = os.path.join("data", "processed", "new_input_processed.csv")
    images_dir: str = os.path.join("data", "images")
    output_csv: str = os.path.join("data", "processed", "new_input_style_color.csv")
    model_path: str = os.path.join("model", "resnet50", "style_kmeans.pkl")
    image_size: int = 224
    max_rows: int | None = None


def run_extract(cfg: ExtractConfig) -> dict:
    model = load_style_model(cfg.model_path)
    if model is None:
        print(f"未找到风格模型: {cfg.model_path}")
        print("请先运行 `python src/image_style_kmeans_train.py` 训练并保存模型。")
        return {
            "status": "model_not_found",
            "model_path": cfg.model_path,
        }

    os.makedirs(os.path.dirname(cfg.output_csv), exist_ok=True)
    data = collect_image_style_data(
        input_csv=cfg.input_csv,
        images_dir=cfg.images_dir,
        image_size=cfg.image_size,
        max_rows=cfg.max_rows,
    )
    out_df = build_style_output(data=data, kmeans=model)
    out_df.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")

    return {
        "status": "ok",
        "input_rows": data.input_rows,
        "used_rows": len(out_df),
        "missing_images": data.missing_images,
        "output_csv": cfg.output_csv,
        "model_path": cfg.model_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加载已训练模型，提取主色与风格")
    parser.add_argument("--input-csv", default=os.path.join("data", "processed", "new_input_processed.csv"))
    parser.add_argument("--images-dir", default=os.path.join("data", "images"))
    parser.add_argument("--output-csv", default=os.path.join("data", "processed", "A_new_input_style_color.csv"))
    parser.add_argument("--model-path", default=os.path.join("model", "resnet50", "style_kmeans.pkl"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    config = ExtractConfig(
        input_csv=args.input_csv,
        images_dir=args.images_dir,
        output_csv=args.output_csv,
        model_path=args.model_path,
        image_size=args.image_size,
        max_rows=args.max_rows,
    )
    report = run_extract(config)
    print("=== 风格与主颜色提取完成 ===")
    for k, v in report.items():
        print(f"{k}: {v}")

