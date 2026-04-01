import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


@dataclass
class ImageStyleData:
    pids: list[str]
    main_colors: list[str]
    features: np.ndarray
    missing_images: int
    input_rows: int


def normalize_pid(value) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def find_image_path(images_dir: str, pid: str) -> str | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
        path = os.path.join(images_dir, f"{pid}{ext}")
        if os.path.exists(path):
            return path
    return None


def rgb_to_basic_name(rgb: np.ndarray) -> str:
    color_map = {
        "black": np.array([0, 0, 0]),
        "white": np.array([255, 255, 255]),
        "gray": np.array([128, 128, 128]),
        "red": np.array([220, 20, 60]),
        "green": np.array([34, 139, 34]),
        "blue": np.array([30, 144, 255]),
        "yellow": np.array([255, 215, 0]),
        "orange": np.array([255, 140, 0]),
        "purple": np.array([138, 43, 226]),
        "pink": np.array([255, 105, 180]),
        "brown": np.array([139, 69, 19]),
        "beige": np.array([245, 245, 220]),
    }
    best = min(color_map.items(), key=lambda kv: np.linalg.norm(rgb - kv[1]))
    return best[0]


def estimate_main_color(image: Image.Image) -> str:
    arr = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float32).reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(arr)
    center = kmeans.cluster_centers_[np.bincount(labels).argmax()]
    return rgb_to_basic_name(center)


class ResNet50FeatureExtractor:
    def __init__(self, image_size: int = 224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def extract(self, image: Image.Image) -> np.ndarray:
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(x).cpu().numpy().reshape(-1)


def collect_image_style_data(
    input_csv: str,
    images_dir: str,
    image_size: int = 224,
    max_rows: int | None = None,
) -> ImageStyleData:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"找不到输入文件: {input_csv}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"找不到图片目录: {images_dir}")

    df = pd.read_csv(input_csv)
    if "pid" not in df.columns:
        raise ValueError("输入数据缺少 pid 字段。")
    if max_rows is not None:
        df = df.head(max_rows).copy()

    extractor = ResNet50FeatureExtractor(image_size=image_size)
    pids: list[str] = []
    main_colors: list[str] = []
    feature_vecs: list[np.ndarray] = []
    missing_images = 0

    for raw_pid in tqdm(df["pid"].values, desc="Extract image features"):
        pid = normalize_pid(raw_pid)
        image_path = find_image_path(images_dir, pid)
        if image_path is None:
            missing_images += 1
            continue
        image = Image.open(image_path).convert("RGB")
        pids.append(pid)
        main_colors.append(estimate_main_color(image))
        feature_vecs.append(extractor.extract(image))

    if not feature_vecs:
        raise RuntimeError("没有可用图片样本，请检查 pid 与 data/images。")

    return ImageStyleData(
        pids=pids,
        main_colors=main_colors,
        features=np.vstack(feature_vecs),
        missing_images=missing_images,
        input_rows=len(df),
    )


def choose_best_kmeans(features: np.ndarray, k_min: int, k_max: int, random_state: int) -> KMeans:
    if len(features) < 2:
        return KMeans(n_clusters=1, random_state=random_state, n_init=10).fit(features)

    best_model = None
    best_score = -1.0
    upper = min(k_max, len(features) - 1)
    lower = max(2, k_min)
    for k in range(lower, upper + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(features)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is not None:
        return best_model

    # fallback: 样本太少或分布退化时，至少给一个可用模型
    return KMeans(n_clusters=min(max(2, k_min), len(features)), random_state=random_state, n_init=10).fit(features)


def save_style_model(kmeans: KMeans, model_path: str, random_state: int):
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    artifact = {
        "type": "resnet50_style_kmeans",
        "kmeans": kmeans,
        "n_clusters": int(kmeans.n_clusters),
        "feature_dim": int(kmeans.cluster_centers_.shape[1]),
        "random_state": random_state,
    }
    joblib.dump(artifact, model_path)


def load_style_model(model_path: str) -> KMeans | None:
    if not os.path.exists(model_path):
        return None
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        model = obj.get("kmeans")
    else:
        model = obj
    if model is None or not hasattr(model, "predict"):
        raise ValueError(f"模型文件格式错误: {model_path}")
    return model


def build_style_output(data: ImageStyleData, kmeans: KMeans) -> pd.DataFrame:
    labels = kmeans.predict(data.features)
    out_df = pd.DataFrame(
        {
            "pid": data.pids,
            "main_color": data.main_colors,
            "style": [f"style_{x}" for x in labels],
        }
    )
    return out_df
