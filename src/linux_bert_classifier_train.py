import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linux-optimized BERT multi-output classifier training"
    )
    parser.add_argument(
        "--project-root",
        default="/data/cloude_storage/PythonProject",
        help="Project root on Linux server",
    )
    parser.add_argument(
        "--csv-path",
        default="data/processed/processed.csv",
        help="Path (relative to project-root or absolute) to processed CSV",
    )
    parser.add_argument(
        "--model-name",
        default="model/bert-base-uncased",
        help="Local pretrained bert path or huggingface model id",
    )
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Dataloader workers, tune based on CPU cores",
    )
    parser.add_argument(
        "--save-dir",
        default="model/linux_bert_classifier",
        help="Directory (relative to project-root or absolute) to save model artifacts",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/linux_bert_classifier",
        help="Directory to save training metrics and plots",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    def parse_params(param_str: str) -> Tuple[str, str]:
        parts = str(param_str).split(", ")
        category = parts[0] if len(parts) > 0 else "unknown"
        sub_category = parts[1] if len(parts) > 1 else "unknown"
        return category, sub_category

    df[["category", "sub_category"]] = df["parameters"].apply(
        lambda x: pd.Series(parse_params(x))
    )
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["category"] = df["category"].fillna("unknown")
    df["sub_category"] = df["sub_category"].fillna("unknown")
    return df


def encode_labels(
    df: pd.DataFrame,
    category_encoder: LabelEncoder = None,
    sub_encoder: LabelEncoder = None,
) -> Tuple[LabelEncoder, LabelEncoder]:
    if category_encoder is None:
        category_encoder = LabelEncoder()
        df["category_id"] = category_encoder.fit_transform(df["category"])
    else:
        df["category_id"] = category_encoder.transform(df["category"])

    if sub_encoder is None:
        sub_encoder = LabelEncoder()
        df["sub_category_id"] = sub_encoder.fit_transform(df["sub_category"])
    else:
        df["sub_category_id"] = sub_encoder.transform(df["sub_category"])

    return category_encoder, sub_encoder


class ProductDataset(Dataset):
    def __init__(
        self,
        texts: np.ndarray,
        category_ids: np.ndarray,
        sub_ids: np.ndarray,
        tokenizer: BertTokenizer,
        max_len: int,
    ):
        self.texts = texts
        self.category_ids = category_ids
        self.sub_ids = sub_ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "category_label": torch.tensor(self.category_ids[idx], dtype=torch.long),
            "sub_label": torch.tensor(self.sub_ids[idx], dtype=torch.long),
        }


class MultiOutputBertClassifier(torch.nn.Module):
    def __init__(self, bert_model: BertModel, num_categories: int, num_sub_categories: int):
        super().__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.1)
        self.category_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_categories
        )
        self.sub_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_sub_categories
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        category_logits = self.category_classifier(pooled_output)
        sub_logits = self.sub_classifier(pooled_output)
        return category_logits, sub_logits


def evaluate(model, val_loader, device, loss_fn):
    model.eval()
    val_loss = 0.0
    all_cat_preds, all_cat_trues = [], []
    all_sub_preds, all_sub_trues = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cat_labels = batch["category_label"].to(device)
            sub_labels = batch["sub_label"].to(device)

            cat_logits, sub_logits = model(input_ids, attention_mask)
            loss_cat = loss_fn(cat_logits, cat_labels)
            loss_sub = loss_fn(sub_logits, sub_labels)
            val_loss += (loss_cat + loss_sub).item()

            all_cat_preds.extend(torch.argmax(cat_logits, dim=1).cpu().numpy())
            all_cat_trues.extend(cat_labels.cpu().numpy())
            all_sub_preds.extend(torch.argmax(sub_logits, dim=1).cpu().numpy())
            all_sub_trues.extend(sub_labels.cpu().numpy())

    avg_val_loss = val_loss / max(len(val_loader), 1)
    cat_acc = accuracy_score(all_cat_trues, all_cat_preds)
    sub_acc = accuracy_score(all_sub_trues, all_sub_preds)
    return avg_val_loss, cat_acc, sub_acc, all_cat_trues, all_cat_preds, all_sub_trues, all_sub_preds


def save_plots(metrics_df: pd.DataFrame, report_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="train_loss")
    plt.plot(metrics_df["epoch"], metrics_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["epoch"], metrics_df["category_acc"], label="category_acc")
    plt.plot(metrics_df["epoch"], metrics_df["sub_acc"], label="sub_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / "val_accuracy_curve.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(args.project_root)
    csv_path = resolve_path(project_root, args.csv_path)
    model_name = str(resolve_path(project_root, args.model_name))
    save_dir = resolve_path(project_root, args.save_dir)
    report_dir = resolve_path(project_root, args.report_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(csv_path)
    print(f"Total samples: {len(df)}")
    train_df, val_df = train_test_split(df, test_size=args.test_size, random_state=args.seed)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    category_encoder, sub_encoder = encode_labels(train_df)
    encode_labels(val_df, category_encoder, sub_encoder)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_encoder = BertModel.from_pretrained(model_name)
    model = MultiOutputBertClassifier(
        bert_encoder, len(category_encoder.classes_), len(sub_encoder.classes_)
    )

    train_dataset = ProductDataset(
        train_df["text"].values,
        train_df["category_id"].values,
        train_df["sub_category_id"].values,
        tokenizer,
        args.max_len,
    )
    val_dataset = ProductDataset(
        val_df["text"].values,
        val_df["category_id"].values,
        val_df["sub_category_id"].values,
        tokenizer,
        args.max_len,
    )

    pin_memory = torch.cuda.is_available()
    workers = max(0, args.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float("inf")
    best_epoch = -1
    history = []

    checkpoint_path = save_dir / "best_classifier.pth"
    category_encoder_path = save_dir / "category_encoder.pkl"
    sub_encoder_path = save_dir / "sub_encoder.pkl"

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")

        for batch in progress:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            cat_labels = batch["category_label"].to(device, non_blocking=True)
            sub_labels = batch["sub_label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                cat_logits, sub_logits = model(input_ids, attention_mask)
                loss_cat = loss_fn(cat_logits, cat_labels)
                loss_sub = loss_fn(sub_logits, sub_labels)
                loss = loss_cat + loss_sub

            total_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(len(train_loader), 1)
        (
            val_loss,
            cat_acc,
            sub_acc,
            cat_true,
            cat_pred,
            sub_true,
            sub_pred,
        ) = evaluate(model, val_loader, device, loss_fn)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "category_acc": cat_acc,
                "sub_acc": sub_acc,
            }
        )
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, category_acc={cat_acc:.4f}, sub_acc={sub_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
            joblib.dump(category_encoder, category_encoder_path)
            joblib.dump(sub_encoder, sub_encoder_path)

            cat_report = classification_report(
                cat_true,
                cat_pred,
                labels=np.arange(len(category_encoder.classes_)),
                target_names=category_encoder.classes_,
                digits=4,
                zero_division=0,
            )
            sub_report = classification_report(
                sub_true,
                sub_pred,
                labels=np.arange(len(sub_encoder.classes_)),
                target_names=sub_encoder.classes_,
                digits=4,
                zero_division=0,
            )
            (report_dir / "best_category_report.txt").write_text(cat_report, encoding="utf-8")
            (report_dir / "best_subcategory_report.txt").write_text(sub_report, encoding="utf-8")

    metrics_df = pd.DataFrame(history)
    metrics_df.to_csv(report_dir / "epoch_metrics.csv", index=False, encoding="utf-8")
    save_plots(metrics_df, report_dir)

    run_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "args": vars(args),
        "paths": {
            "project_root": str(project_root),
            "csv_path": str(csv_path),
            "model_name": model_name,
            "save_dir": str(save_dir),
            "report_dir": str(report_dir),
            "best_checkpoint": str(checkpoint_path),
        },
        "dataset": {
            "total_samples": int(len(df)),
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "num_categories": int(len(category_encoder.classes_)),
            "num_sub_categories": int(len(sub_encoder.classes_)),
        },
    }
    (report_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nTraining complete.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Reports saved in: {report_dir}")


if __name__ == "__main__":
    main()
