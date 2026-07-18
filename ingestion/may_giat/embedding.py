import json
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

DATASET = "may_giat"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_embedded.json"
MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 128


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu để embedding")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    # E5 yêu cầu prefix passage cho dữ liệu được index.
    texts = [f"passage: {item['text']}" for item in data]
    with torch.inference_mode():
        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
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
