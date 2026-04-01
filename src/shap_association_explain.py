from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from transformers import BertForSequenceClassification, BertForTokenClassification, BertTokenizer

from image_style_module import build_style_output, collect_image_style_data, load_style_model


FEATURE_COLS = ["category", "sub_category", "brand", "fabric", "color", "main_color", "style"]
TARGET_COLS = ["sales", "repeat_rate", "average_rating"]


@dataclass
class Config:
    project_root: str = r"C:\Users\86155\Desktop\PythonProject"
    username: str = "ggpower"
    raw_csv_name: str = "1.csv"

    bert_base_path: str = r"C:\Users\86155\Desktop\PythonProject\model\bert-base-uncased"
    bert_classifier_path: str = r"C:\Users\86155\Desktop\PythonProject\model\bert_classifier\best_classifier.pth"
    bert_category_encoder_path: str = r"C:\Users\86155\Desktop\PythonProject\model\bert_classifier\category_encoder.pkl"
    bert_sub_encoder_path: str = r"C:\Users\86155\Desktop\PythonProject\model\bert_classifier\sub_encoder.pkl"
    bert_ner_path: str = r"C:\Users\86155\Desktop\PythonProject\model\bert_ner\best_ner_model.pth"
    style_kmeans_path: str = r"C:\Users\86155\Desktop\PythonProject\model\resnet50\A_style_kmeans.pkl"

    regression_model_path: str = r"C:\Users\86155\Desktop\PythonProject\model\regression_model\best_regression_model.pth"
    tree_model_path: str = r"C:\Users\86155\Desktop\PythonProject\model\decision_tree_model\best_decision_tree_model.pth"
    background_csv: str = r"C:\Users\86155\Desktop\PythonProject\data\processed\A_final_input.csv"
    background_size: int | None = None
    random_state: int = 42
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


LABEL_LIST = ["O", "B-material", "I-material", "B-color", "I-color", "B-brand", "I-brand"]
id2label = {idx: label for idx, label in enumerate(LABEL_LIST)}


def _user_paths(cfg: Config):
    raw_csv_path = os.path.join(cfg.project_root, "user", cfg.username, "raw", cfg.raw_csv_name)
    image_dir = os.path.join(cfg.project_root, "user", cfg.username, "images")
    raw_stem = Path(cfg.raw_csv_name).stem
    return raw_csv_path, image_dir, raw_stem


def _build_output_dir(cfg: Config, pid: str, raw_stem: str) -> str:
    _ = raw_stem
    out_dir = os.path.join(cfg.project_root, "user", cfg.username, "shap", str(pid))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _load_user_raw(raw_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"未找到用户上传CSV: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    required = ["pid", "title", "description"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"用户CSV缺少字段: {miss}")
    df["pid"] = df["pid"].astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["text"] = (df["title"] + " " + df["description"]).str.strip()
    return df


def _clean_wordpiece(tokens: List[str]) -> str:
    out = ""
    for token in tokens:
        if token.startswith("##"):
            out += token[2:]
        elif out:
            out += " " + token
        else:
            out = token
    return out.strip()


def _decode_entities(tokens: List[str], label_ids: List[int]) -> Dict[str, str]:
    bucket = {"material": [], "color": [], "brand": []}
    current_tokens = []
    current_field = None
    for token, label_id in zip(tokens, label_ids):
        label = id2label.get(label_id, "O")
        if token in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        if label == "O":
            if current_tokens and current_field:
                bucket[current_field].append(_clean_wordpiece(current_tokens))
            current_tokens, current_field = [], None
            continue
        if label.startswith("B-"):
            if current_tokens and current_field:
                bucket[current_field].append(_clean_wordpiece(current_tokens))
            current_field = {"material": "material", "color": "color", "brand": "brand"}.get(label[2:], None)
            current_tokens = [token] if current_field else []
        elif label.startswith("I-") and current_field is not None:
            current_tokens.append(token)
    if current_tokens and current_field:
        bucket[current_field].append(_clean_wordpiece(current_tokens))
    return {k: ", ".join(v).strip() for k, v in bucket.items()}


def _predict_text_features(df: pd.DataFrame, cfg: Config, device: torch.device) -> pd.DataFrame:
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_base_path)
    category_encoder = joblib.load(cfg.bert_category_encoder_path)
    sub_encoder = joblib.load(cfg.bert_sub_encoder_path)

    backbone = BertForSequenceClassification.from_pretrained(cfg.bert_base_path, num_labels=2).bert
    clf_model = MultiOutputBertClassifier(
        bert_model=backbone,
        num_categories=len(category_encoder.classes_),
        num_sub_categories=len(sub_encoder.classes_),
    ).to(device)
    clf_model.load_state_dict(torch.load(cfg.bert_classifier_path, map_location=device))
    clf_model.eval()

    ner_model = BertForTokenClassification.from_pretrained(cfg.bert_base_path, num_labels=len(LABEL_LIST)).to(device)
    ner_model.load_state_dict(torch.load(cfg.bert_ner_path, map_location=device))
    ner_model.eval()

    category_preds = []
    sub_preds = []
    fabrics = []
    colors = []
    brands = []
    with torch.no_grad():
        for text in df["text"].values:
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=cfg.max_text_len,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            cat_logits, sub_logits = clf_model(input_ids, attention_mask)
            category_preds.append(category_encoder.inverse_transform([torch.argmax(cat_logits, dim=1).item()])[0])
            sub_preds.append(sub_encoder.inverse_transform([torch.argmax(sub_logits, dim=1).item()])[0])

            logits = ner_model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy().tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy().tolist())
            entities = _decode_entities(tokens, pred_ids)
            fabrics.append(entities.get("material", ""))
            colors.append(entities.get("color", ""))
            brands.append(entities.get("brand", ""))

    out = df.copy()
    out["category"] = category_preds
    out["sub_category"] = sub_preds
    out["fabric"] = fabrics
    out["color"] = colors
    out["brand"] = brands
    return out


def _predict_image_features(df: pd.DataFrame, cfg: Config, raw_csv_path: str, image_dir: str) -> pd.DataFrame:
    style_model = load_style_model(cfg.style_kmeans_path)
    if style_model is None:
        raise FileNotFoundError(f"未找到风格模型: {cfg.style_kmeans_path}")

    image_data = collect_image_style_data(
        input_csv=raw_csv_path,
        images_dir=image_dir,
        image_size=cfg.image_size,
    )
    image_df = build_style_output(image_data, style_model)
    image_df["pid"] = image_df["pid"].astype(str)
    base = df.copy()
    base["pid"] = base["pid"].astype(str)
    merged = base.merge(image_df, on="pid", how="inner")
    if len(merged) == 0:
        raise RuntimeError("用户图片未命中，无法进行后续预测和解释。")
    return merged


def _load_background_features(cfg: Config) -> pd.DataFrame:
    if not os.path.isfile(cfg.background_csv):
        raise FileNotFoundError(f"未找到背景数据: {cfg.background_csv}")
    bg = pd.read_csv(cfg.background_csv)
    if "average_rating" not in bg.columns and "avarage_rating" in bg.columns:
        bg = bg.copy()
        bg["average_rating"] = bg["avarage_rating"]
    miss = [c for c in FEATURE_COLS if c not in bg.columns]
    if miss:
        raise ValueError(f"背景数据缺少特征列: {miss}")
    if cfg.background_size is not None and cfg.background_size > 0 and len(bg) > cfg.background_size:
        bg = bg.sample(n=cfg.background_size, random_state=cfg.random_state)
    for c in FEATURE_COLS:
        bg[c] = bg[c].fillna("unknown").astype(str).replace("", "unknown")
    return bg[FEATURE_COLS].reset_index(drop=True)


def _predict_with_model(model_obj, X_df: pd.DataFrame) -> np.ndarray:
    if hasattr(model_obj, "predict"):
        return model_obj.predict(X_df)
    if isinstance(model_obj, dict) and "model" in model_obj:
        m = model_obj["model"]
        feature_names = m["feature_names"]
        x_dummies = pd.get_dummies(X_df, columns=FEATURE_COLS, dummy_na=False, dtype=float)
        x_dummies = x_dummies.reindex(columns=feature_names, fill_value=0.0)
        x_mat = x_dummies.to_numpy(dtype=float)
        return x_mat @ m["coef"] + m["intercept"]
    raise ValueError("模型格式不支持，无法预测。")


def _save_multi_plots(
    est,
    Xt_eval,
    Xt_bg,
    feat_names: List[str],
    target_name: str,
    model_tag: str,
    out_dir: str,
):
    def _to_dense_2d(x):
        if hasattr(x, "toarray"):
            return x.toarray().astype(float)
        return np.asarray(x, dtype=float)

    def _to_dense_row(x):
        return _to_dense_2d(x)[0].ravel()

    Xt_bg_dense = _to_dense_2d(Xt_bg)
    Xt_eval_dense = _to_dense_2d(Xt_eval)

    if isinstance(est, DecisionTreeRegressor):
        # 避免 tree_path_dependent 的叶子覆盖限制
        try:
            explainer = shap.TreeExplainer(est, data=Xt_bg_dense, feature_perturbation="interventional")
        except Exception:
            try:
                explainer = shap.TreeExplainer(est)
            except Exception:
                explainer = shap.Explainer(est.predict, Xt_bg_dense)
    elif isinstance(est, LinearRegression):
        explainer = shap.LinearExplainer(est, Xt_bg_dense)
    else:
        explainer = shap.Explainer(est.predict, Xt_bg_dense)

    try:
        sv = explainer(Xt_eval_dense)
    except Exception:
        # 某些版本在调用阶段仍可能触发 ExplainerError，统一兜底
        fallback = shap.Explainer(est.predict, Xt_bg_dense)
        sv = fallback(Xt_eval_dense)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, Xt_eval_dense, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"summary_{model_tag}_{target_name}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    shap.plots.bar(sv, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"bar_{model_tag}_{target_name}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(sv[0], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"waterfall_{model_tag}_{target_name}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    force_features = _to_dense_row(Xt_eval_dense)
    force_html = shap.force_plot(
        sv.base_values[0],
        sv.values[0],
        features=force_features,
        feature_names=feat_names,
    )
    shap.save_html(os.path.join(out_dir, f"force_{model_tag}_{target_name}.html"), force_html)


def run(cfg: Config):
    raw_csv_path, image_dir, raw_stem = _user_paths(cfg)
    user_df = _load_user_raw(raw_csv_path)
    pid = str(user_df.iloc[0]["pid"])
    out_dir = _build_output_dir(cfg, pid, raw_stem)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"用户: {cfg.username}, pid: {pid}, raw: {raw_csv_path}")

    # 1) 特征提取（不保存）
    txt_df = _predict_text_features(user_df, cfg, device)
    enriched = _predict_image_features(txt_df, cfg, raw_csv_path, image_dir)
    X_user = enriched[FEATURE_COLS].copy()
    for c in FEATURE_COLS:
        X_user[c] = X_user[c].fillna("unknown").astype(str).replace("", "unknown")

    # 2) 加载模型并预测
    reg_obj = joblib.load(cfg.regression_model_path)
    tree_obj = joblib.load(cfg.tree_model_path)
    pred_reg = _predict_with_model(reg_obj, X_user)
    pred_tree = _predict_with_model(tree_obj, X_user)

    print("\n=== 用户样本预测结果 ===")
    print(f"pid: {pid}")
    print(
        "回归模型预测: "
        f"sales={pred_reg[0,0]:.4f}, repeat_rate={pred_reg[0,1]:.6f}, average_rating={pred_reg[0,2]:.4f}"
    )
    print(
        "决策树模型预测: "
        f"sales={pred_tree[0,0]:.4f}, repeat_rate={pred_tree[0,1]:.6f}, average_rating={pred_tree[0,2]:.4f}"
    )

    # 3) SHAP 可解释性（背景取 A_final_input）
    X_bg = _load_background_features(cfg)

    if hasattr(reg_obj, "named_steps"):
        reg_prep = reg_obj.named_steps["prep"]
        reg_model = reg_obj.named_steps["model"]
        Xt_bg_reg = reg_prep.transform(X_bg)
        Xt_user_reg = reg_prep.transform(X_user)
        reg_feat_names = reg_prep.get_feature_names_out().tolist()
        for i, est in enumerate(reg_model.estimators_):
            _save_multi_plots(est, Xt_user_reg, Xt_bg_reg, reg_feat_names, TARGET_COLS[i], "regression", out_dir)
    else:
        print("[warn] 回归模型不是 sklearn Pipeline，跳过回归 SHAP 图。")

    if hasattr(tree_obj, "named_steps"):
        tree_prep = tree_obj.named_steps["prep"]
        tree_model = tree_obj.named_steps["model"]
        Xt_bg_tree = tree_prep.transform(X_bg)
        Xt_user_tree = tree_prep.transform(X_user)
        tree_feat_names = tree_prep.get_feature_names_out().tolist()
        for i, est in enumerate(tree_model.estimators_):
            _save_multi_plots(est, Xt_user_tree, Xt_bg_tree, tree_feat_names, TARGET_COLS[i], "tree", out_dir)
    else:
        print("[warn] 决策树模型不是 sklearn Pipeline，跳过决策树 SHAP 图。")

    print(f"\nSHAP 可视化输出目录: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="模拟系统应用流程的 SHAP 可解释性分析")
    parser.add_argument("--username", default="ggpower")
    parser.add_argument("--raw-csv-name", default="1.csv", help="用户 raw 目录下文件名")
    parser.add_argument("--project-root", default=r"C:\Users\86155\Desktop\PythonProject")
    parser.add_argument("--background-size", type=int, default=0, help="背景样本数，0表示使用全部背景数据")
    args = parser.parse_args()

    cfg = Config(
        project_root=args.project_root,
        username=args.username,
        raw_csv_name=args.raw_csv_name,
        background_size=None if args.background_size == 0 else args.background_size,
    )
    run(cfg)


if __name__ == "__main__":
    main()
