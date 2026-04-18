import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from transformers import BertForTokenClassification, BertTokenizerFast


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
ENTITY_KEY = {"BRAND": "brand", "COLOR": "color", "FABRIC": "material"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick extraction test for BERT NER model")
    parser.add_argument(
        "--checkpoint-path",
        default="model/bert_ner_agent/best_ner_new.pth",
        help="Path to model .pth checkpoint",
    )
    parser.add_argument(
        "--model-name",
        default="C:/Users/86155/Desktop/PythonProject/model/bert-base-uncased",
        help="Tokenizer/base model path",
    )
    parser.add_argument(
        "--input-jsonl",
        default="output/agent_bio_annotations.cleaned.jsonl",
        help="Input JSONL containing title/description and optional gold fields",
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample records instead of taking first N",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            title = str(row.get("title", "")).strip()
            desc = str(row.get("description", "")).strip()
            if not title and not desc:
                continue
            records.append(row)
    return records


def unique_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        x = x.strip()
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


def decode_entities(
    text: str,
    pred_ids: List[int],
    offset_mapping: List[List[int]],
) -> Dict[str, List[str]]:
    entities = {"brand": [], "color": [], "material": []}
    cur_type = None
    cur_start = None
    cur_end = None

    def flush():
        nonlocal cur_type, cur_start, cur_end
        if cur_type is not None and cur_start is not None and cur_end is not None:
            ent_text = text[cur_start:cur_end].strip()
            key = ENTITY_KEY.get(cur_type)
            if key and ent_text:
                entities[key].append(ent_text)
        cur_type, cur_start, cur_end = None, None, None

    for idx, pid in enumerate(pred_ids):
        start, end = offset_mapping[idx]
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(pid), "O")
        if label == "O":
            flush()
            continue
        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            flush()
            cur_type = ent_type
            cur_start = start
            cur_end = end
        elif prefix == "I":
            if cur_type == ent_type and cur_start is not None:
                cur_end = end
            else:
                flush()
                cur_type = ent_type
                cur_start = start
                cur_end = end
    flush()

    for k in entities:
        entities[k] = unique_keep_order(entities[k])
    return entities


def predict_one(
    model,
    tokenizer: BertTokenizerFast,
    text: str,
    max_len: int,
    device: torch.device,
) -> Dict[str, List[str]]:
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offsets = encoding["offset_mapping"][0].tolist()

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    return decode_entities(text=text, pred_ids=pred_ids, offset_mapping=offsets)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_path = Path(args.input_jsonl)
    ckpt_path = Path(args.checkpoint_path)
    records = load_records(input_path)
    if not records:
        raise ValueError(f"No valid records in {input_path}")

    if args.random_sample:
        selected = random.sample(records, k=min(args.num_samples, len(records)))
    else:
        selected = records[: args.num_samples]

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"Showing {len(selected)} samples:\n")

    for row in selected:
        title = str(row.get("title", "")).strip()
        description = str(row.get("description", "")).strip()
        text = f"{title}. {description}".strip()
        pred = predict_one(model, tokenizer, text, args.max_len, device)
        gold = row.get("gold_entities_from_parameter", {})
        out = {
            "id": row.get("id", ""),
            "title": title,
            "pred": pred,
            "gold": gold,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print("-" * 80)


if __name__ == "__main__":
    main()
