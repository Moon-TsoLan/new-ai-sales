import os
from dataclasses import dataclass
from typing import Dict, List

import joblib
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertForTokenClassification, BertTokenizer

from image_style_module import build_style_output, collect_image_style_data, load_style_model


@dataclass
class Config:
    input_csv: str = r"C:\Users\86155\Desktop\PythonProject\data\processed\new_input_processed.csv"
    image_dir: str = r"C:\Users\86155\Desktop\PythonProject\data\images"
    output_csv: str = r"C:\Users\86155\Desktop\PythonProject\data\process\A_final_input.csv"
    image_style_color_csv: str = r"C:\Users\86155\Desktop\PythonProject\data\process\A_new_input_style_color.csv"

    bert_base_path: str = r"C:\Users\86155\Desktop\PythonProject\model\bert-base-uncased"
    bert_classifier_path: str = r"C:\Users\86155\Desktop\PythonProject\model\bert_classifier\best_classifier.pth"
    bert_category_encoder_path: str = r"C:\Users\86155\Desktop\PythonProject\model\bert_classifier\category_encoder.pkl"
    bert_sub_encoder_path: str = r"C:\Users\86155\Desktop\PythonProject\model\bert_classifier\sub_encoder.pkl"
    bert_ner_path: str = r"C:\Users\86155\Desktop\PythonProject\model\bert_ner\best_ner_model.pth"

    style_kmeans_path: str = r"C:\Users\86155\Desktop\PythonProject\model\resnet50\A_style_kmeans.pkl"
    max_text_len: int = 256
    image_size: int = 224


class MultiOutputBertClassifier(nn.Module):
    def __init__(self, bert_model, num_categories: int, num_sub_categories: int):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.category_classifier = nn.Linear(self.bert.config.hidden_size, num_categories)
        self.sub_classifier = nn.Linear(self.bert.config.hidden_size, num_sub_categories)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        return self.category_classifier(pooled_output), self.sub_classifier(pooled_output)


LABEL_LIST = [
    "O",
    "B-material",
    "I-material",
    "B-color",
    "I-color",
    "B-brand",
    "I-brand",
]
id2label = {idx: label for idx, label in enumerate(LABEL_LIST)}


def ensure_dirs(config: Config):
    os.makedirs(os.path.dirname(config.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(config.image_style_color_csv), exist_ok=True)


def load_base_data(config: Config) -> pd.DataFrame:
    if not os.path.exists(config.input_csv):
        raise FileNotFoundError(f"未找到输入文件: {config.input_csv}")
    df = pd.read_csv(config.input_csv)
    required = ["pid", "title", "description"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"输入文件缺少字段: {missing}")
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["text"] = (df["title"] + " " + df["description"]).str.strip()
    return df


def clean_wordpiece(tokens: List[str]) -> str:
    text = ""
    for token in tokens:
        if token.startswith("##"):
            text += token[2:]
        elif text:
            text += " " + token
        else:
            text = token
    return text.strip()


def decode_ner_entities(tokens: List[str], label_ids: List[int]) -> Dict[str, str]:
    bucket = {"material": [], "color": [], "brand": []}
    current_tokens = []
    current_field = None
    for token, label_id in zip(tokens, label_ids):
        label = id2label.get(label_id, "O")
        if token in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        if label == "O":
            if current_tokens and current_field:
                bucket[current_field].append(clean_wordpiece(current_tokens))
            current_tokens, current_field = [], None
            continue
        if label.startswith("B-"):
            if current_tokens and current_field:
                bucket[current_field].append(clean_wordpiece(current_tokens))
            current_field = {"material": "material", "color": "color", "brand": "brand"}.get(label[2:], None)
            current_tokens = [token] if current_field else []
        elif label.startswith("I-") and current_field is not None:
            current_tokens.append(token)
    if current_tokens and current_field:
        bucket[current_field].append(clean_wordpiece(current_tokens))
    return {k: ", ".join(v).strip() for k, v in bucket.items()}


def predict_text_features(df: pd.DataFrame, config: Config, device: torch.device) -> pd.DataFrame:
    tokenizer = BertTokenizer.from_pretrained(config.bert_base_path)

    if not os.path.exists(config.bert_classifier_path):
        raise FileNotFoundError(f"未找到 bert 分类器模型: {config.bert_classifier_path}")
    if not os.path.exists(config.bert_category_encoder_path) or not os.path.exists(config.bert_sub_encoder_path):
        raise FileNotFoundError("未找到 bert 分类器编码器文件。")
    if not os.path.exists(config.bert_ner_path):
        raise FileNotFoundError(f"未找到 bert_ner 模型: {config.bert_ner_path}")

    category_encoder = joblib.load(config.bert_category_encoder_path)
    sub_encoder = joblib.load(config.bert_sub_encoder_path)

    backbone = BertForSequenceClassification.from_pretrained(config.bert_base_path, num_labels=2).bert
    clf_model = MultiOutputBertClassifier(
        bert_model=backbone,
        num_categories=len(category_encoder.classes_),
        num_sub_categories=len(sub_encoder.classes_),
    ).to(device)
    clf_model.load_state_dict(torch.load(config.bert_classifier_path, map_location=device))
    clf_model.eval()

    ner_model = BertForTokenClassification.from_pretrained(config.bert_base_path, num_labels=len(LABEL_LIST)).to(device)
    ner_model.load_state_dict(torch.load(config.bert_ner_path, map_location=device))
    ner_model.eval()

    category_preds = []
    sub_category_preds = []
    fabrics = []
    colors = []
    brands = []

    with torch.no_grad():
        for text in tqdm(df["text"].values, desc="Text feature infer"):
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=config.max_text_len,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            cat_logits, sub_logits = clf_model(input_ids, attention_mask)
            category_preds.append(category_encoder.inverse_transform([torch.argmax(cat_logits, dim=1).item()])[0])
            sub_category_preds.append(sub_encoder.inverse_transform([torch.argmax(sub_logits, dim=1).item()])[0])

            logits = ner_model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy().tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy().tolist())
            entities = decode_ner_entities(tokens, pred_ids)
            fabrics.append(entities.get("material", ""))
            colors.append(entities.get("color", ""))
            brands.append(entities.get("brand", ""))

    out = df.copy()
    out["category"] = category_preds
    out["sub_category"] = sub_category_preds
    out["fabric"] = fabrics
    out["color"] = colors
    out["brand"] = brands
    return out


def extract_image_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    style_model = load_style_model(config.style_kmeans_path)
    if style_model is None:
        raise FileNotFoundError(
            f"未找到 KMeans 模型: {config.style_kmeans_path}。"
            "请先运行 src/image_style_kmeans_train.py 生成 A_style_kmeans.pkl。"
        )

    image_data = collect_image_style_data(
        input_csv=config.input_csv,
        images_dir=config.image_dir,
        image_size=config.image_size,
    )
    image_df = build_style_output(image_data, style_model)
    image_df.to_csv(config.image_style_color_csv, index=False, encoding="utf-8-sig")
    print(f"图片匹配统计：总样本={image_data.input_rows}，命中={len(image_df)}，未命中={image_data.missing_images}")
    print(f"图像特征文件已保存: {config.image_style_color_csv}")

    # 按需求：图片丢失样本不记录 -> 用 inner merge 仅保留命中图片的 pid
    base_df = df.copy()
    base_df["pid"] = base_df["pid"].astype(str)
    image_df["pid"] = image_df["pid"].astype(str)
    return base_df.merge(image_df, on="pid", how="inner")


def main():
    config = Config()
    ensure_dirs(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    data = load_base_data(config)
    data = predict_text_features(data, config, device)
    final_df = extract_image_features(data, config)
    final_df.to_csv(config.output_csv, index=False, encoding="utf-8-sig")

    print(f"最终特征数据已保存: {config.output_csv}")
    print(f"最终样本数(仅保留命中图片): {len(final_df)}")


if __name__ == "__main__":
    main()

