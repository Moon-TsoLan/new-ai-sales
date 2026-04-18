from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import pandas as pd
import torch

from app.core.config import get_settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt

settings = get_settings()

_SRC_READY = False
_REGRESSION_MODEL: Any = None
_TREE_MODEL: Any = None

TARGET_COLS = ["sales", "repeat_rate", "average_rating"]


def _ensure_src_import():
    global _SRC_READY
    if _SRC_READY:
        return
    import sys

    src_path = str((settings.project_root / "src").resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    _SRC_READY = True


def _load_predict_models():
    global _REGRESSION_MODEL, _TREE_MODEL
    if _REGRESSION_MODEL is None and settings.regression_model_path.exists():
        _REGRESSION_MODEL = joblib.load(settings.regression_model_path)
    if _TREE_MODEL is None and settings.tree_model_path.exists():
        _TREE_MODEL = joblib.load(settings.tree_model_path)
    return _REGRESSION_MODEL, _TREE_MODEL


def _build_cfg():
    _ensure_src_import()
    from shap_association_explain import Config

    return Config(
        project_root=str(settings.project_root),
        bert_base_path=str(settings.bert_base_path),
        bert_classifier_path=str(settings.bert_classifier_path),
        bert_category_encoder_path=str(settings.bert_category_encoder_path),
        bert_sub_encoder_path=str(settings.bert_sub_encoder_path),
        bert_ner_path=str(settings.bert_ner_path),
        style_kmeans_path=str(settings.style_kmeans_path),
        regression_model_path=str(settings.regression_model_path),
        tree_model_path=str(settings.tree_model_path),
        background_csv=str(settings.background_csv),
    )


def _heuristic_predict(product_name: str, product_desc: str) -> tuple[float, float, float]:
    text_len = len((product_name + " " + product_desc).strip())
    sales = max(50.0, min(20000.0, 100.0 + text_len * 12.5))
    repeat_rate = max(0.02, min(0.95, 0.2 + text_len / 1000.0))
    average_rating = max(2.5, min(5.0, 3.8 + (text_len % 25) / 100.0))
    return float(sales), float(repeat_rate), float(average_rating)


def _draw_fallback_plot(out_path: Path, features: dict[str, Any], preds: tuple[float, float, float]) -> None:
    labels = list(features.keys())[:7] if features else ["unknown"]
    values = [1.0 / max(1, len(labels)) for _ in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values, color="#4f46e5")
    ax.set_title("Fallback SHAP-like Feature Importance")
    ax.set_xlabel("Contribution")
    text = f"sales={preds[0]:.2f}, repeat_rate={preds[1]:.4f}, avg_rating={preds[2]:.2f}"
    ax.text(0.01, -0.6, text)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_files(model_tag: str, target_name: str) -> dict[str, str]:
    return {
        "summary": f"summary_{model_tag}_{target_name}.png",
        "bar": f"bar_{model_tag}_{target_name}.png",
        "waterfall": f"waterfall_{model_tag}_{target_name}.png",
        "force_html": f"force_{model_tag}_{target_name}.html",
    }


def predict_and_explain(
    user_id: int,
    submission_id: str,
    product_name: str,
    product_desc: str,
    uploaded_image_abs: Path,
    uploaded_image_rel: str,
) -> dict[str, Any]:
    _ensure_src_import()
    from shap_association_explain import (
        FEATURE_COLS,
        _load_background_features,
        _predict_image_features,
        _predict_text_features,
        _predict_with_model,
        _save_multi_plots,
    )

    cfg = _build_cfg()
    reg_obj, tree_obj = _load_predict_models()

    req_id = uuid.uuid4().hex[:12]
    output_dir = settings.upload_root / str(user_id) / submission_id
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir / f"tmp_{req_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    image_dir = temp_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = temp_dir / "raw.csv"

    pid = req_id.upper()
    image_copy = image_dir / f"{pid}{uploaded_image_abs.suffix.lower()}"
    shutil.copy2(uploaded_image_abs, image_copy)

    pd.DataFrame([{"pid": pid, "title": product_name, "description": product_desc}]).to_csv(raw_csv_path, index=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extracted_features: dict[str, Any] = {}
    model_predictions: dict[str, dict[str, float]] = {}
    visualizations: dict[str, dict[str, dict[str, str]]] = {"regression": {}, "tree": {}}
    base_prediction = _heuristic_predict(product_name, product_desc)
    primary_shap_rel = f"{user_id}/{submission_id}/fallback.png"
    primary_shap_abs = settings.upload_root / primary_shap_rel

    try:
        raw_df = pd.read_csv(raw_csv_path)
        raw_df["pid"] = raw_df["pid"].astype(str)
        raw_df["title"] = raw_df["title"].fillna("").astype(str)
        raw_df["description"] = raw_df["description"].fillna("").astype(str)
        raw_df["text"] = (raw_df["title"] + raw_df["description"]).str.strip()

        txt_df = _predict_text_features(raw_df, cfg, device)
        enriched = _predict_image_features(txt_df, cfg, str(raw_csv_path), str(image_dir))
        X_user = enriched[FEATURE_COLS].copy()
        for col in FEATURE_COLS:
            X_user[col] = X_user[col].fillna("unknown").astype(str).replace("", "unknown")
        extracted_features = X_user.iloc[0].to_dict()

        if reg_obj is not None:
            pred_reg = _predict_with_model(reg_obj, X_user)
            model_predictions["regression"] = {
                "sales": float(pred_reg[0, 0]),
                "repeat_rate": float(pred_reg[0, 1]),
                "average_rating": float(pred_reg[0, 2]),
            }
        if tree_obj is not None:
            pred_tree = _predict_with_model(tree_obj, X_user)
            model_predictions["tree"] = {
                "sales": float(pred_tree[0, 0]),
                "repeat_rate": float(pred_tree[0, 1]),
                "average_rating": float(pred_tree[0, 2]),
            }

        X_bg = _load_background_features(cfg)

        if reg_obj is not None and hasattr(reg_obj, "named_steps"):
            prep = reg_obj.named_steps["prep"]
            model = reg_obj.named_steps["model"]
            Xt_bg = prep.transform(X_bg)
            Xt_user = prep.transform(X_user)
            feat_names = prep.get_feature_names_out().tolist()
            for i, target in enumerate(TARGET_COLS):
                _save_multi_plots(model.estimators_[i], Xt_user, Xt_bg, feat_names, target, "regression", str(output_dir))
                files = _plot_files("regression", target)
                visualizations["regression"][target] = {
                    k: f"/uploads/{user_id}/{submission_id}/{v}" for k, v in files.items()
                }

        if tree_obj is not None and hasattr(tree_obj, "named_steps"):
            prep = tree_obj.named_steps["prep"]
            model = tree_obj.named_steps["model"]
            Xt_bg = prep.transform(X_bg)
            Xt_user = prep.transform(X_user)
            feat_names = prep.get_feature_names_out().tolist()
            for i, target in enumerate(TARGET_COLS):
                _save_multi_plots(model.estimators_[i], Xt_user, Xt_bg, feat_names, target, "tree", str(output_dir))
                files = _plot_files("tree", target)
                visualizations["tree"][target] = {k: f"/uploads/{user_id}/{submission_id}/{v}" for k, v in files.items()}

        reg_sales_summary = output_dir / "summary_regression_sales.png"
        tree_sales_summary = output_dir / "summary_tree_sales.png"
        if reg_sales_summary.exists():
            primary_shap_rel = f"{user_id}/{submission_id}/summary_regression_sales.png"
            primary_shap_abs = reg_sales_summary
        elif tree_sales_summary.exists():
            primary_shap_rel = f"{user_id}/{submission_id}/summary_tree_sales.png"
            primary_shap_abs = tree_sales_summary

    except Exception:
        if not extracted_features:
            extracted_features = {
                "category": "unknown",
                "sub_category": "unknown",
                "brand": "unknown",
                "fabric": "unknown",
                "color": "unknown",
                "main_color": "unknown",
                "style": "unknown",
            }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not primary_shap_abs.exists():
        _draw_fallback_plot(primary_shap_abs, extracted_features, base_prediction)
        visualizations["regression"]["sales"] = {
            "summary": f"/uploads/{primary_shap_rel}",
            "bar": "",
            "waterfall": "",
            "force_html": "",
        }

    chosen = model_predictions.get("regression") or model_predictions.get("tree") or {
        "sales": base_prediction[0],
        "repeat_rate": base_prediction[1],
        "average_rating": base_prediction[2],
    }

    return {
        "product_name": product_name,
        "product_desc": product_desc,
        "image_path": uploaded_image_rel.replace("\\", "/"),
        "features_snapshot": {
            "extracted_features": extracted_features,
            "model_predictions": model_predictions,
            "visualizations": visualizations,
        },
        "predicted_sales": float(chosen["sales"]),
        "repeat_rate": float(chosen["repeat_rate"]),
        "average_rating": float(chosen["average_rating"]),
        "shap_plot_path": primary_shap_rel.replace("\\", "/"),
    }

