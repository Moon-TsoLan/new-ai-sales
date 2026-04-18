"""
Agent-based BIO labeling for product NER.

Input:
- data/processed/processed.csv
- uses title + description for extraction
- reads material/color/brand reference from parameter(s)

Output:
- JSONL with extracted entities and token-level BIO tags
- summary JSON with quick quality metrics against parameter reference
"""
import argparse
import csv
import difflib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


ENTITY_KEYS = ("brand", "color", "material")
BIO_PREFIX = {"brand": "BRAND", "color": "COLOR", "material": "FABRIC"}
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# ===== In-code defaults (can still be overridden by CLI) =====
# Fill your real key here if you prefer hardcoded config.
DEFAULT_MINIMAX_API_KEY = "sk-api-kIQA39BXZv-mAE6VKNwP3Ft3o67SN6v7_95epfNGCM9WM2snEElpreimFqDqr1NdE4K0sboj1NpHUSx8XKQzUAXgRRX_9NLiuy7R52JVpoZZEA_fHkI2ku4"
DEFAULT_MAX_SAMPLES = 10
DEFAULT_MAX_RETRIES = 1
MATERIAL_STOPWORDS = {
    "rich",
    "combed",
    "friendly",
    "fabric",
    "premium",
    "quality",
    "lightweight",
    "heavy",
    "soft",
    "ultra",
    "fine",
    "pure",
    "made",
    "blend",
    "blended",
    "percent",
}
MATERIAL_TOKEN_SYNONYMS = {
    "poly": "polyester",
    "elastane": "spandex",
    "lycra": "spandex",
    "rayon": "viscose",
}


@dataclass
class EvalCounter:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent BIO labeling pipeline")
    parser.add_argument(
        "--input-csv",
        default="data/processed/processed.csv",
        help="Input CSV path with title/description/parameter(s) columns",
    )
    parser.add_argument(
        "--output-jsonl",
        default="output/agent_bio_annotations.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--progress-file",
        default="output/agent_bio_progress.txt",
        help="Checkpoint file for processed row ids",
    )
    parser.add_argument(
        "--error-log",
        default="output/agent_bio_errors.log",
        help="Error log file",
    )
    parser.add_argument(
        "--summary-file",
        default="output/agent_bio_eval_summary.json",
        help="Evaluation summary file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Max rows to process in this run (for quick validation)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based start row index in CSV (e.g. 10 means start from the 11th row)",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Take one row every N rows after start-index (e.g. 10 means 1/10 sampling)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep between API calls to reduce rate limits",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AGENT_BIO_MODEL", "MiniMax-M2.5"),
        help="Model name for extraction",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.minimax.chat/v1"),
        help="Optional OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="MINIMAX_API_KEY",
        help="Environment variable name storing API key",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_MINIMAX_API_KEY,
        help="API key string. If set, it overrides environment variables.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite output/progress files before running",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retry times per row when parsing/API fails",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" ,;:/|-_")
    return value


def split_entity_values(raw: str) -> List[str]:
    if not raw:
        return []
    parts = [normalize_text(x) for x in re.split(r"[,/;|]", raw)]
    return [p for p in parts if p]


def split_raw_values(raw: str) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in re.split(r"[,/;|]", str(raw)) if x.strip()]


def parse_parameter_field(parameter_value: str) -> Dict[str, str]:
    """
    Expected source format (typical):
    category, sub_category, material, color[, color2 ...], brand
    """
    if parameter_value is None:
        return {k: "" for k in ENTITY_KEYS}

    parts = [p.strip() for p in str(parameter_value).split(",")]
    parts = [p for p in parts if p is not None]
    if len(parts) < 5:
        return {k: "" for k in ENTITY_KEYS}

    material = parts[2].strip()
    brand = parts[-1].strip()
    color = ", ".join(p.strip() for p in parts[3:-1] if p.strip())
    return {"brand": brand, "color": color, "material": material}


def _tokenize_for_match(value: str, field: str) -> set:
    text = normalize_text(value)
    tokens = re.findall(r"[a-z0-9]+", text)
    result = set()
    for token in tokens:
        if token.isdigit():
            continue
        token = MATERIAL_TOKEN_SYNONYMS.get(token, token)
        if field == "material":
            if token in MATERIAL_STOPWORDS:
                continue
            if len(token) <= 1:
                continue
        result.add(token)
    return result


def _similarity(pred: str, cand: str, field: str) -> float:
    p = normalize_text(pred)
    c = normalize_text(cand)
    if not p or not c:
        return 0.0
    if p == c:
        return 1.0

    seq = difflib.SequenceMatcher(None, p, c).ratio()
    pt = _tokenize_for_match(p, field)
    ct = _tokenize_for_match(c, field)
    jaccard = len(pt & ct) / len(pt | ct) if (pt or ct) else 0.0
    contains = 0.0
    if p in c or c in p:
        contains = 0.88
    if field == "brand":
        if p.startswith(c) or c.startswith(p):
            contains = max(contains, 0.9)
    if field == "material" and pt and ct and (pt & ct):
        contains = max(contains, 0.86)

    return max(contains, 0.55 * seq + 0.45 * jaccard)


def build_catalog_vocab(rows: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, int]]]:
    vocab = {field: {} for field in ENTITY_KEYS}
    freq = {field: {} for field in ENTITY_KEYS}
    for row in rows:
        parameter_value = row.get("parameter", row.get("parameters", ""))
        gold = parse_parameter_field(parameter_value)
        for field in ENTITY_KEYS:
            for raw_value in split_raw_values(gold.get(field, "")):
                norm = normalize_text(raw_value)
                if norm and norm not in vocab[field]:
                    vocab[field][norm] = raw_value.strip()
                if norm:
                    freq[field][norm] = freq[field].get(norm, 0) + 1
    return vocab, freq


def canonicalize_pred_entities(
    pred_entities: Dict[str, List[str]],
    catalog_vocab: Dict[str, Dict[str, str]],
    catalog_freq: Dict[str, Dict[str, int]],
) -> Dict[str, List[str]]:
    thresholds = {"brand": 0.78, "color": 0.85, "material": 0.62}
    result: Dict[str, List[str]] = {field: [] for field in ENTITY_KEYS}

    for field in ENTITY_KEYS:
        vocab_norm_to_raw = catalog_vocab.get(field, {})
        vocab_norms = list(vocab_norm_to_raw.keys())
        seen = set()

        for value in pred_entities.get(field, []):
            for part in split_raw_values(value):
                norm_part = normalize_text(part)
                if not norm_part:
                    continue

                if norm_part in vocab_norm_to_raw:
                    mapped_norm = norm_part
                    # Generic material rule:
                    # If a base material is very often represented as "X Blend" in catalog,
                    # map to the dominant blend variant for better consistency.
                    if field == "material":
                        pred_tokens = _tokenize_for_match(norm_part, field)
                        base_freq = catalog_freq[field].get(norm_part, 0)
                        best_blend_norm = ""
                        best_blend_freq = -1
                        for cand_norm in vocab_norms:
                            cand_tokens = _tokenize_for_match(cand_norm, field)
                            if "blend" not in cand_norm:
                                continue
                            if pred_tokens and pred_tokens.issubset(cand_tokens):
                                cand_freq = catalog_freq[field].get(cand_norm, 0)
                                if cand_freq > best_blend_freq:
                                    best_blend_freq = cand_freq
                                    best_blend_norm = cand_norm
                        if best_blend_norm and best_blend_freq >= max(2 * base_freq, 2):
                            mapped_norm = best_blend_norm
                    mapped = vocab_norm_to_raw[mapped_norm]
                else:
                    best_score = 0.0
                    best_norm = ""
                    for cand_norm in vocab_norms:
                        score = _similarity(norm_part, cand_norm, field)
                        if score > best_score:
                            best_score = score
                            best_norm = cand_norm
                    if best_norm and best_score >= thresholds[field]:
                        mapped = vocab_norm_to_raw[best_norm]
                    else:
                        mapped = part.strip()

                mapped_norm = normalize_text(mapped)
                if mapped_norm and mapped_norm not in seen:
                    seen.add(mapped_norm)
                    result[field].append(mapped)
    return result


def build_material_base_lexicon(catalog_freq: Dict[str, Dict[str, int]]) -> Dict[str, str]:
    token_freq: Dict[str, int] = {}
    for mat_norm, freq in catalog_freq.get("material", {}).items():
        for token in _tokenize_for_match(mat_norm, "material"):
            if token in MATERIAL_STOPWORDS:
                continue
            token_freq[token] = token_freq.get(token, 0) + freq

    # Keep only recurring material roots to avoid one-off noise.
    lexicon = {}
    for token, freq in token_freq.items():
        if freq >= 2:
            lexicon[token] = token.title()
    return lexicon


def map_to_material_base(values: List[str], material_base_lexicon: Dict[str, str]) -> set:
    base_set = set()
    for value in values:
        for part in split_raw_values(value):
            for token in _tokenize_for_match(part, "material"):
                if token in material_base_lexicon:
                    base_set.add(normalize_text(material_base_lexicon[token]))
    return base_set


def safe_json_loads(content: str) -> Dict:
    text = (content or "").strip()
    if not text:
        raise json.JSONDecodeError("Empty response content", text, 0)

    # Remove think/reasoning wrappers often returned by some providers.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # Prefer fenced JSON block if present.
    if "```json" in text.lower():
        lower = text.lower()
        start = lower.find("```json")
        end = lower.find("```", start + 7)
        if end != -1:
            text = text[start + 7 : end].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    # Direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: locate the first JSON object in mixed text.
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("No valid JSON object found", text, 0)


def create_client(api_key: str, base_url: Optional[str]) -> OpenAI:
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def clean_entity_dict(raw: Dict) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {k: [] for k in ENTITY_KEYS}
    if not isinstance(raw, dict):
        return result
    for key in ENTITY_KEYS:
        value = raw.get(key, [])
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            value = []
        cleaned = []
        seen = set()
        for item in value:
            norm = normalize_text(str(item))
            if norm and norm not in seen:
                seen.add(norm)
                cleaned.append(str(item).strip())
        result[key] = cleaned
    return result


def extract_entities_with_agent(
    client: OpenAI,
    model: str,
    title: str,
    description: str,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    prompt = f"""
Extract product attributes from title + description.
Return JSON only:
{{
  "mention": {{"brand": [], "color": [], "material": []}},
  "canonical": {{"brand": [], "color": [], "material": []}}
}}

Rules:
1) mention = exact text spans from title/description.
2) canonical = normalized catalog values.
3) brand canonical uses root form when obvious (Yorker -> York).
4) material canonical normalizes verbose phrase (100% rich combed cotton -> Cotton/Cotton Blend).
5) color canonical keeps concise style (Dark Blue, Multicolor).
6) Use [] when unknown.

title: {title}
description: {description}
""".strip()

    def call_and_parse(user_prompt: str) -> Dict:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=600,
        )
        return safe_json_loads(response.choices[0].message.content)

    try:
        parsed = call_and_parse(prompt)
    except json.JSONDecodeError:
        strict_prompt = f"""
Output EXACTLY one JSON object. No markdown, no explanation.
Schema:
{{"mention":{{"brand":[],"color":[],"material":[]}},"canonical":{{"brand":[],"color":[],"material":[]}}}}
title: {title}
description: {description}
""".strip()
        parsed = call_and_parse(strict_prompt)

    if any(k in parsed for k in ENTITY_KEYS):
        mention = clean_entity_dict(parsed)
        canonical = clean_entity_dict(parsed)
        return mention, canonical

    mention = clean_entity_dict(parsed.get("mention", {}))
    canonical = clean_entity_dict(parsed.get("canonical", {}))
    for field in ENTITY_KEYS:
        if not canonical[field]:
            canonical[field] = mention[field]
    return mention, canonical


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def find_all_spans(text: str, phrase: str) -> List[Tuple[int, int]]:
    phrase = phrase.strip()
    if not phrase:
        return []
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def select_non_overlapping_spans(
    text: str, entities: Dict[str, List[str]]
) -> List[Tuple[str, int, int]]:
    candidates: List[Tuple[str, int, int]] = []
    for field, values in entities.items():
        for value in values:
            for start, end in find_all_spans(text, value):
                candidates.append((field, start, end))

    # Prefer longer mentions, then earlier positions.
    candidates.sort(key=lambda x: (-(x[2] - x[1]), x[1]))

    selected: List[Tuple[str, int, int]] = []
    occupied = [False] * max(len(text), 1)
    for field, start, end in candidates:
        if start < 0 or end <= start:
            continue
        if any(occupied[i] for i in range(start, min(end, len(text)))):
            continue
        for i in range(start, min(end, len(text))):
            occupied[i] = True
        selected.append((field, start, end))
    return sorted(selected, key=lambda x: x[1])


def entities_to_bio(text: str, entities: Dict[str, List[str]]) -> List[Dict[str, str]]:
    tokens = tokenize_with_spans(text)
    spans = select_non_overlapping_spans(text, entities)
    labels = ["O"] * len(tokens)

    for field, start_char, end_char in spans:
        indices = []
        for i, (_, t_start, t_end) in enumerate(tokens):
            if t_start >= start_char and t_end <= end_char:
                indices.append(i)
        if not indices:
            continue
        labels[indices[0]] = f"B-{BIO_PREFIX[field]}"
        for idx in indices[1:]:
            labels[idx] = f"I-{BIO_PREFIX[field]}"

    return [{"word": token, "tag": tag} for (token, _, _), tag in zip(tokens, labels)]


def load_checkpoint(progress_file: Path) -> set:
    if not progress_file.exists():
        return set()
    with progress_file.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_checkpoint(progress_file: Path, row_id: str) -> None:
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(f"{row_id}\n")


def row_id_from_row(row: Dict[str, str], index: int) -> str:
    for key in ("id", "ID", "pid", "uuid"):
        value = row.get(key)
        if value:
            return str(value)
    return f"row_{index}"


def set_from_field_value(value: str) -> set:
    return set(split_entity_values(value))


def set_from_pred_values(values: List[str]) -> set:
    merged = []
    for item in values:
        merged.extend(split_entity_values(str(item)))
    return set(merged)


def score_sets(pred_set: set, gold_set: set, counter: EvalCounter) -> None:
    counter.tp += len(pred_set & gold_set)
    counter.fp += len(pred_set - gold_set)
    counter.fn += len(gold_set - pred_set)


def run() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_jsonl)
    progress_path = Path(args.progress_file)
    error_path = Path(args.error_log)
    summary_path = Path(args.summary_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite_output:
        for p in (output_path, progress_path, error_path, summary_path):
            if p.exists():
                p.unlink()

    api_key = args.api_key.strip()
    if api_key == "your_minimax_api_key":
        api_key = ""
    if not api_key:
        api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        api_key = os.getenv("MINIMAX_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "Missing API key. Set DEFAULT_MINIMAX_API_KEY in code, use --api-key, or set MINIMAX_API_KEY."
        )
    client = create_client(api_key, args.base_url)

    processed_ids = load_checkpoint(progress_path)
    print(f"Loaded {len(processed_ids)} processed ids.")

    with input_path.open("r", encoding="utf-8-sig", newline="") as f_in:
        rows = list(csv.DictReader(f_in))
    catalog_vocab, catalog_freq = build_catalog_vocab(rows)
    material_base_lexicon = build_material_base_lexicon(catalog_freq)

    start_index = max(args.start_index, 0)
    stride = max(args.sample_stride, 1)
    sampled_rows = rows[start_index::stride]
    if args.max_samples > 0:
        rows_to_process = sampled_rows[: args.max_samples]
    else:
        rows_to_process = sampled_rows
    last_possible_index = len(rows) - 1 if rows else 0
    print(
        f"Total rows loaded: {len(rows)}. "
        f"Sampling from index {start_index} to {last_possible_index} "
        f"with stride={stride}; selected {len(rows_to_process)} rows."
    )

    eval_by_field = {k: EvalCounter() for k in ENTITY_KEYS}
    exact_match_count = 0
    evaluated_count = 0
    success_count = 0

    with output_path.open("a", encoding="utf-8") as f_out:
        for i, row in enumerate(tqdm(rows_to_process, desc="agent-bio")):
            row_id = row_id_from_row(row, i)
            if row_id in processed_ids:
                continue
            
            title = (row.get("title") or "").strip()
            description = (row.get("description") or "").strip()
            full_text = f"{title}. {description}".strip()
            parameter_value = row.get("parameter", row.get("parameters", ""))
            gold = parse_parameter_field(parameter_value)

            try:
                mention_pred: Dict[str, List[str]] = {k: [] for k in ENTITY_KEYS}
                canonical_pred: Dict[str, List[str]] = {k: [] for k in ENTITY_KEYS}
                last_err: Optional[Exception] = None
                for attempt in range(args.max_retries + 1):
                    try:
                        mention_pred, canonical_pred = extract_entities_with_agent(
                            client=client,
                            model=args.model,
                            title=title,
                            description=description,
                        )
                        last_err = None
                        break
                    except Exception as exc:
                        last_err = exc
                        if attempt < args.max_retries:
                            time.sleep(0.5)
                if last_err is not None:
                    raise last_err

                canonical_pred = canonicalize_pred_entities(
                    canonical_pred,
                    catalog_vocab,
                    catalog_freq,
                )
                bio = entities_to_bio(full_text, mention_pred)

                per_field_equal = True
                for field in ENTITY_KEYS:
                    if field == "material":
                        pred_set = map_to_material_base(
                            canonical_pred.get(field, []), material_base_lexicon
                        )
                        gold_set = map_to_material_base(
                            split_raw_values(gold.get(field, "")), material_base_lexicon
                        )
                    else:
                        pred_set = set_from_pred_values(canonical_pred.get(field, []))
                        gold_set = set_from_field_value(gold.get(field, ""))
                    score_sets(pred_set, gold_set, eval_by_field[field])
                    if pred_set != gold_set:
                        per_field_equal = False

                evaluated_count += 1
                if per_field_equal:
                    exact_match_count += 1

                result = {
                    "id": row_id,
                    "title": title,
                    "description": description,
                    "text": full_text,
                    "pred_entities": mention_pred,
                    "pred_entities_mention": mention_pred,
                    "pred_entities_canonical": canonical_pred,
                    "gold_entities_from_parameter": gold,
                    "material_base_pred": sorted(
                        map_to_material_base(
                            canonical_pred.get("material", []), material_base_lexicon
                        )
                    ),
                    "material_base_gold": sorted(
                        map_to_material_base(
                            split_raw_values(gold.get("material", "")),
                            material_base_lexicon,
                        )
                    ),
                    "bio_labels": bio,
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                append_checkpoint(progress_path, row_id)
                processed_ids.add(row_id)
                success_count += 1
                time.sleep(args.sleep_seconds)

            except Exception as exc:
                with error_path.open("a", encoding="utf-8") as f_err:
                    f_err.write(f"{row_id}\t{type(exc).__name__}\t{exc}\n")

    summary = {
        "model": args.model,
        "eval_on": "brand/color: canonical vs gold; material: base-material mapping vs gold",
        "catalog_vocab_sizes": {
            field: len(catalog_vocab[field]) for field in ENTITY_KEYS
        },
        "material_base_lexicon_size": len(material_base_lexicon),
        "input_csv": str(input_path),
        "processed_count": success_count,
        "evaluated_count": evaluated_count,
        "exact_match_ratio_all_fields": (
            exact_match_count / evaluated_count if evaluated_count else 0.0
        ),
        "field_metrics": {
            field: {
                "precision": counter.precision(),
                "recall": counter.recall(),
                "f1": counter.f1(),
                "tp": counter.tp,
                "fp": counter.fp,
                "fn": counter.fn,
            }
            for field, counter in eval_by_field.items()
        },
        "output_jsonl": str(output_path),
        "error_log": str(error_path),
    }
    with summary_path.open("w", encoding="utf-8") as f_summary:
        json.dump(summary, f_summary, ensure_ascii=False, indent=2)

    print("\nRun finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()