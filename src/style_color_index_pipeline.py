import argparse
import os
from dataclasses import dataclass

from image_style_module import (
    build_style_output,
    choose_best_kmeans,
    collect_image_style_data,
    load_style_model,
    save_style_model,
)


@dataclass
class PipelineConfig:
    input_csv: str
    images_dir: str
    output_csv: str
    model_path: str
    image_size: int = 224
    k_min: int = 2
    k_max: int = 10
    random_state: int = 42
    max_rows: int | None = None


def build_index(cfg: PipelineConfig) -> dict:
    os.makedirs(os.path.dirname(cfg.output_csv), exist_ok=True)
    data = collect_image_style_data(
        input_csv=cfg.input_csv,
        images_dir=cfg.images_dir,
        image_size=cfg.image_size,
        max_rows=cfg.max_rows,
    )
    kmeans = choose_best_kmeans(
        features=data.features,
        k_min=cfg.k_min,
        k_max=cfg.k_max,
        random_state=cfg.random_state,
    )
    save_style_model(kmeans=kmeans, model_path=cfg.model_path, random_state=cfg.random_state)
    out_df = build_style_output(data=data, kmeans=kmeans)
    out_df.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")

    return {
        "mode": "build-index",
        "input_rows": data.input_rows,
        "used_rows": len(out_df),
        "missing_images": data.missing_images,
        "n_clusters": int(kmeans.n_clusters),
        "output_csv": cfg.output_csv,
        "model_path": cfg.model_path,
    }


def assign_to_existing_index(cfg: PipelineConfig) -> dict:
    model = load_style_model(cfg.model_path)
    if model is None:
        print(f"未找到风格模型: {cfg.model_path}")
        print("请先运行训练模式（--mode build-index）或 src/image_style_kmeans_train.py。")
        return {
            "mode": "assign",
            "status": "model_not_found",
            "model_path": cfg.model_path,
        }
    data = collect_image_style_data(
        input_csv=cfg.input_csv,
        images_dir=cfg.images_dir,
        image_size=cfg.image_size,
        max_rows=cfg.max_rows,
    )
    out_df = build_style_output(data=data, kmeans=model)
    out_df.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")

    return {
        "mode": "assign",
        "status": "ok",
        "input_rows": data.input_rows,
        "used_rows": len(out_df),
        "missing_images": data.missing_images,
        "output_csv": cfg.output_csv,
        "model_path": cfg.model_path,
    }


def main():
    parser = argparse.ArgumentParser(description="ResNet50 风格索引构建与增量分配")
    parser.add_argument("--mode", choices=["build-index", "assign"], required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--images-dir", default=os.path.join("data", "images"))
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model-path", default=os.path.join("model", "style_index", "style_index.pkl"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    cfg = PipelineConfig(
        input_csv=args.input_csv,
        images_dir=args.images_dir,
        output_csv=args.output_csv,
        model_path=args.model_path,
        image_size=args.image_size,
        k_min=args.k_min,
        k_max=args.k_max,
        max_rows=args.max_rows,
    )

    if args.mode == "build-index":
        report = build_index(cfg)
    else:
        report = assign_to_existing_index(cfg)

    print("=== style pipeline done ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

