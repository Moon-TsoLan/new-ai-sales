import argparse
import os
from dataclasses import dataclass

from image_style_module import collect_image_style_data, choose_best_kmeans, save_style_model


@dataclass
class TrainConfig:
    input_csv: str
    image_dir: str
    model_path: str
    image_size: int = 224
    k_min: int = 2
    k_max: int = 10
    random_state: int = 42
    max_rows: int | None = None


def run_train(cfg: TrainConfig):
    data = collect_image_style_data(
        input_csv=cfg.input_csv,
        images_dir=cfg.image_dir,
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

    print("=== image style train done ===")
    print(f"输入样本: {data.input_rows}")
    print(f"命中图片: {len(data.pids)}")
    print(f"未命中图片: {data.missing_images}")
    print(f"风格簇数量(k): {kmeans.n_clusters}")
    print(f"模型文件: {cfg.model_path}")


def main():
    parser = argparse.ArgumentParser(description="训练图片风格模型并仅保存模型文件")
    parser.add_argument("--input-csv", default=os.path.join("data", "processed", "new_input_processed.csv"))
    parser.add_argument("--image-dir", default=os.path.join("data", "images"))
    parser.add_argument("--model-path", default=os.path.join("model", "resnet50", "A_style_kmeans.pkl"))
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        input_csv=args.input_csv,
        image_dir=args.image_dir,
        model_path=args.model_path,
        image_size=args.image_size,
        k_min=args.k_min,
        k_max=args.k_max,
        max_rows=args.max_rows,
    )
    run_train(cfg)


if __name__ == "__main__":
    main()

