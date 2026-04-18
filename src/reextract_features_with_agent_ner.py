import argparse
import csv
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
    parser = argparse.ArgumentParser(
        description="Re-extract brand/color/fabric with bert_ner_agent model"
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/A_final_input.csv",
        help="Input CSV path (must contain title, description, brand, color, fabric)",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/A_final_input_bert_ner.csv",
        help="Output CSV path with overwritten brand/color/fabric columns",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="model/bert_ner_agent/best_ner_new.pth",
        help="Trained .pth checkpoint path",
    )
    parser.add_argument(
        "--model-name",
        default="C:/Users/86155/Desktop/PythonProject/model/bert-base-uncased",
        help="Tokenizer/base model path",
    )
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Debug mode: only process first N rows; 0 means all rows",
    )
    parser.add_argument(
        "--fallback-to-old",
        action="store_true",
        help="Keep old brand/color/fabric when prediction is empty",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write results back to input file path",
    )
    return parser.parse_args()


def unique_keep_order(values: List[str]) -> List[str]:
    out = []
    seen = set()
    for v in values:
        s = str(v).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def normalize_color_values(colors: List[str]) -> List[str]:
    merged = " ".join([c.lower() for c in colors])
    out = list(colors)
    if "multi color" in merged or "multi-color" in merged:
        out = [c for c in out if c.lower() not in {"multi", "color"}]
        out.append("Multicolor")
    return unique_keep_order(out)


def normalize_entity_output(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    brand = unique_keep_order(entities.get("brand", []))
    color = normalize_color_values(unique_keep_order(entities.get("color", [])))
    material = unique_keep_order(entities.get("material", []))

    # Keep concise material if model gives many fragments.
    if material:
        material = sorted(material, key=lambda x: len(x), reverse=True)
        material = [material[0]]
    if brand:
        brand = [brand[0]]

    return {"brand": brand, "color": color, "material": material}


def decode_entities_from_prediction(
    text: str, pred_ids: List[int], offsets: List[List[int]]
) -> Dict[str, List[str]]:
    entities = {"brand": [], "color": [], "material": []}
    cur_type = None
    cur_start = None
    cur_end = None

    def flush():
        nonlocal cur_type, cur_start, cur_end
        if cur_type is not None and cur_start is not None and cur_end is not None:
            span = text[cur_start:cur_end].strip()
            key = ENTITY_KEY.get(cur_type)
            if key and span:
                entities[key].append(span)
        cur_type, cur_start, cur_end = None, None, None

    for i, pid in enumerate(pred_ids):
        start, end = offsets[i]
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
    return normalize_entity_output(entities)


def to_joined(values: List[str]) -> str:
    return ", ".join(unique_keep_order(values))


def predict_batch(
    model,
    tokenizer: BertTokenizerFast,
    texts: List[str],
    max_len: int,
    device: torch.device,
) -> List[Dict[str, List[str]]]:
    encoding = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offsets_batch = encoding["offset_mapping"].tolist()

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_ids_batch = torch.argmax(logits, dim=-1).cpu().tolist()

    outputs = []
    for text, pred_ids, offsets in zip(texts, pred_ids_batch, offsets_batch):
        outputs.append(decode_entities_from_prediction(text, pred_ids, offsets))
    return outputs


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = input_csv if args.inplace else Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    state = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    required = {"title", "description", "brand", "color", "fabric"}
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    batch_size = max(1, args.batch_size)
    out_rows = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [
            f"{(r.get('title') or '').strip()}. {(r.get('description') or '').strip()}"
            for r in batch
        ]
        pred_entities = predict_batch(model, tokenizer, texts, args.max_len, device)

        for row, pred in zip(batch, pred_entities):
            fabric_old = (row.get("fabric") or "").strip()
            color_old = (row.get("color") or "").strip()
            brand_old = (row.get("brand") or "").strip()

            fabric_new = to_joined(pred.get("material", []))
            color_new = to_joined(pred.get("color", []))
            brand_new = to_joined(pred.get("brand", []))

            if args.fallback_to_old:
                if not fabric_new:
                    fabric_new = fabric_old
                if not color_new:
                    color_new = color_old
                if not brand_new:
                    brand_new = brand_old

            row["fabric"] = fabric_new
            row["color"] = color_new
            row["brand"] = brand_new
            out_rows.append(row)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        f"Done. Wrote {len(out_rows)} rows to {output_csv}. "
        f"Device={device}, fallback_to_old={args.fallback_to_old}, inplace={args.inplace}"
    )


if __name__ == "__main__":
    main()
