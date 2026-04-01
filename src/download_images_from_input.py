import argparse
import ast
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


URL_RE = re.compile(r"https?://[^\s'\",]+")


def parse_first_image_url(images_value: object) -> str | None:
    """从 images 字段中提取第一张图片 URL。"""
    if images_value is None:
        return None

    s = str(images_value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    # 常见格式： "['url1', 'url2']"
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)) and parsed:
            first = str(parsed[0]).strip()
            return first if first.startswith(("http://", "https://")) else None
    except Exception:
        pass

    # 兜底：正则提取 URL，取第一个
    matches = URL_RE.findall(s)
    return matches[0] if matches else None


def infer_extension(url: str, content_type: str | None) -> str:
    """优先从 URL 路径推断扩展名，其次从 content-type 推断。"""
    path = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
        if path.endswith(ext):
            return ext

    if content_type:
        ct = content_type.lower()
        if "jpeg" in ct or "jpg" in ct:
            return ".jpg"
        if "png" in ct:
            return ".png"
        if "webp" in ct:
            return ".webp"
        if "gif" in ct:
            return ".gif"
        if "bmp" in ct:
            return ".bmp"
    return ".jpg"


def download_images(
    input_xlsx: str,
    output_dir: str,
    timeout: int = 20,
    overwrite: bool = False,
    max_rows: int | None = None,
) -> dict:
    if not os.path.exists(input_xlsx):
        raise FileNotFoundError(f"找不到输入文件: {input_xlsx}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_xlsx)
    required_cols = {"pid", "images"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"input.xlsx 缺少必须列: {required_cols}，实际列: {set(df.columns)}")

    if max_rows is not None:
        df = df.head(max_rows)

    total = len(df)
    ok = 0
    skipped_exists = 0
    skipped_empty = 0
    failed = 0

    # 用 session 复用连接，下载更稳定
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0 (ImageDownloader/1.0)"})

        for i, row in df.iterrows():
            pid = str(row["pid"]).strip()
            if not pid or pid.lower() in {"nan", "none"}:
                skipped_empty += 1
                continue

            first_url = parse_first_image_url(row["images"])
            if not first_url:
                skipped_empty += 1
                continue

            # 先用 .jpg 检查同名是否已存在（也会检查其它常见后缀）
            existing = None
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
                p = out / f"{pid}{ext}"
                if p.exists():
                    existing = p
                    break

            if existing is not None and not overwrite:
                skipped_exists += 1
                continue

            try:
                resp = session.get(first_url, timeout=timeout)
                resp.raise_for_status()
                ext = infer_extension(first_url, resp.headers.get("Content-Type"))
                save_path = out / f"{pid}{ext}"
                with open(save_path, "wb") as f:
                    f.write(resp.content)
                ok += 1
            except Exception as e:
                failed += 1
                print(f"[FAIL] row={i} pid={pid} url={first_url} err={e}")

    return {
        "total_rows": total,
        "downloaded": ok,
        "skipped_exists": skipped_exists,
        "skipped_empty_or_invalid": skipped_empty,
        "failed": failed,
        "output_dir": str(out),
    }


def main():
    parser = argparse.ArgumentParser(description="从 input.xlsx 的 images 字段下载图片到 data/images。")
    parser.add_argument(
        "--input",
        default=os.path.join("data", "raw", "input.xlsx"),
        help="输入 Excel 路径（默认: data/raw/input.xlsx）",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "images"),
        help="图片输出目录（默认: data/images）",
    )
    parser.add_argument("--timeout", type=int, default=20, help="下载超时时间（秒）")
    parser.add_argument("--overwrite", action="store_true", help="已存在同名文件时覆盖")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="仅处理前 N 行（用于测试）",
    )
    args = parser.parse_args()

    report = download_images(
        input_xlsx=args.input,
        output_dir=args.output_dir,
        timeout=args.timeout,
        overwrite=args.overwrite,
        max_rows=args.max_rows,
    )

    print("=== 下载完成 ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

