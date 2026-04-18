import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
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
        description="Train BERT token-classification from agent BIO labels"
    )
    parser.add_argument(
        "--input-jsonl",
        default="output/agent_bio_annotations.cleaned.jsonl",
        help="Cleaned JSONL produced by clean_agent_annotations.py",
    )
    parser.add_argument(
        "--model-name",
        default="C:/Users/86155/Desktop/PythonProject/model/bert-base-uncased",
        help="Local pretrained bert path or model id",
    )
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        default="model/bert_ner_agent/best_ner_new.pth",
        help="Best checkpoint path",
    )
    return parser.parse_args()


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
    # Backward compatibility with B-material style labels.
    t = t.replace("MATERIAL", "FABRIC")
    if t not in LABEL2ID:
        return "O"
    return t


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


def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
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

    if not all_trues:
        return 0.0, "No valid validation labels."
    f1 = f1_score(all_trues, all_preds)
    report = classification_report(all_trues, all_preds, digits=4)
    return f1, report


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    input_path = Path(args.input_jsonl)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl_samples(input_path)
    if len(samples) < 10:
        raise ValueError(f"Too few samples: {len(samples)}")
    print(f"Loaded {len(samples)} labeled samples from {input_path}")

    train_samples, val_samples = train_test_split(
        samples, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = AgentNerDataset(train_samples, tokenizer, args.max_len)
    val_dataset = AgentNerDataset(val_samples, tokenizer, args.max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * max(total_steps, 1)),
        num_training_steps=max(total_steps, 1),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    best_f1 = 0.0
    patience = 2
    wait = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training {epoch + 1}", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Train loss: {avg_loss:.4f}")

        val_f1, report = evaluate(model, val_loader, device)
        print(report)
        print(f"Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nTraining done. Best Val F1: {best_f1:.4f}")


if __name__ == "__main__":
    train(parse_args())
