"""Embed cooler/freezer semantic documents with multilingual E5."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


DATASET = "tu_mat_tu_dong"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_embedded.json"
MODEL_NAME = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384
BATCH_SIZE = 128


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu tủ mát, tủ đông để embedding")
    texts: list[str] = []
    for index, item in enumerate(data):
        text = str(item.get("text") or "").strip()
        if not text:
            raise ValueError(f"Sản phẩm index {index} thiếu semantic text")
        texts.append(f"passage: {text}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    with torch.inference_mode():
        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
    if vectors.ndim != 2 or vectors.shape != (len(data), VECTOR_SIZE):
        raise ValueError(
            f"Embedding có shape {vectors.shape}; yêu cầu ({len(data)}, {VECTOR_SIZE})"
        )

    for item, vector in zip(data, vectors):
        item["vector"] = vector.astype(float).tolist()

    OUTPUT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã embedding {len(data)} sản phẩm trên {device} -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
