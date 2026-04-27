import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from seqeval.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)


LABEL_LIST = [
    "O",
    "B-BRAND",
    "I-BRAND",
    "B-COLOR",
    "I-COLOR",
    "B-FABRIC",
    "I-FABRIC",
]
LABEL2ID = {x: i for i, x in enumerate(LABEL_LIST)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linux-optimized BERT token-classification training from BIO labels"
    )
    parser.add_argument(
        "--project-root",
        default="/data/cloude_storage/PythonProject",
        help="Project root on Linux server",
    )
    parser.add_argument(
        "--input-jsonl",
        default="output/agent_bio_annotations.cleaned.jsonl",
        help="BIO JSONL data path (relative to project-root or absolute)",
    )
    parser.add_argument(
        "--model-name",
        default="model/bert-base-uncased",
        help="Local pretrained bert path or huggingface model id",
    )
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--save-dir",
        default="model/linux_bert_ner_agent",
        help="Directory for best checkpoint",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/linux_bert_ner_agent",
        help="Directory to save metrics and plots",
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


def normalize_tag(tag: str) -> str:
    if not tag:
        return "O"
    t = str(tag).strip().upper()
    t = t.replace("MATERIAL", "FABRIC")
    return t if t in LABEL2ID else "O"


def load_jsonl_samples(input_path: Path) -> List[Dict]:
    samples = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            bio_labels = row.get("bio_labels", [])
            if not isinstance(bio_labels, list) or not bio_labels:
                continue

            words = []
            tags = []
            for item in bio_labels:
                if not isinstance(item, dict):
                    continue
                word = str(item.get("word", "")).strip()
                if not word:
                    continue
                words.append(word)
                tags.append(normalize_tag(item.get("tag", "O")))

            if words and len(words) == len(tags):
                samples.append({"id": row.get("id", ""), "words": words, "tags": tags})
    return samples


class AgentNerDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer: BertTokenizerFast, max_len: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        words = sample["words"]
        tags = sample["tags"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)

        labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                labels.append(LABEL2ID[tags[word_id]])
            else:
                labels.append(-100)
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def evaluate(model, data_loader, device) -> Tuple[float, str]:
    model.eval()
    all_preds, all_trues = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating", unit="batch"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)

            for i in range(labels.shape[0]):
                pred_seq = []
                true_seq = []
                for j in range(labels.shape[1]):
                    true_id = labels[i, j].item()
                    if true_id == -100:
                        continue
                    pred_seq.append(ID2LABEL[pred_ids[i, j].item()])
                    true_seq.append(ID2LABEL[true_id])
                if true_seq:
                    all_preds.append(pred_seq)
                    all_trues.append(true_seq)

    avg_val_loss = total_loss / max(len(data_loader), 1)
    if not all_trues:
        return 0.0, "No valid validation labels.", avg_val_loss
    f1 = f1_score(all_trues, all_preds)
    report = classification_report(all_trues, all_preds, digits=4)
    return f1, report, avg_val_loss


def save_plots(metrics_df: pd.DataFrame, report_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="train_loss")
    plt.plot(metrics_df["epoch"], metrics_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("NER Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["epoch"], metrics_df["val_f1"], label="val_f1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("NER Validation F1 Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / "val_f1_curve.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(args.project_root)
    input_path = resolve_path(project_root, args.input_jsonl)
    model_name = str(resolve_path(project_root, args.model_name))
    save_dir = resolve_path(project_root, args.save_dir)
    report_dir = resolve_path(project_root, args.report_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl_samples(input_path)
    if len(samples) < 10:
        raise ValueError(f"Too few samples: {len(samples)}")
    print(f"Loaded {len(samples)} labeled samples from {input_path}")

    train_samples, val_samples = train_test_split(
        samples, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = AgentNerDataset(train_samples, tokenizer, args.max_len)
    val_dataset = AgentNerDataset(val_samples, tokenizer, args.max_len)

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

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_f1 = 0.0
    best_epoch = -1
    wait = 0
    metrics = []

    best_model_path = save_dir / "best_ner_new.pth"

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training {epoch + 1}", unit="batch"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            total_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        train_loss = total_loss / max(len(train_loader), 1)
        val_f1, report, val_loss = evaluate(model, val_loader, device)
        print(report)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
            }
        )
        (report_dir / f"classification_report_epoch_{epoch + 1}.txt").write_text(
            report, encoding="utf-8"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            wait = 0
            torch.save(model.state_dict(), best_model_path)
            (report_dir / "best_classification_report.txt").write_text(
                report, encoding="utf-8"
            )
            print(f"Saved best model to {best_model_path}")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(report_dir / "epoch_metrics.csv", index=False, encoding="utf-8")
    save_plots(metrics_df, report_dir)

    run_summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "args": vars(args),
        "paths": {
            "project_root": str(project_root),
            "input_jsonl": str(input_path),
            "model_name": model_name,
            "save_dir": str(save_dir),
            "report_dir": str(report_dir),
            "best_checkpoint": str(best_model_path),
        },
        "dataset": {
            "total_samples": int(len(samples)),
            "train_samples": int(len(train_samples)),
            "val_samples": int(len(val_samples)),
            "label_list": LABEL_LIST,
        },
    }
    (report_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTraining complete.")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Reports saved in: {report_dir}")


if __name__ == "__main__":
    main()
