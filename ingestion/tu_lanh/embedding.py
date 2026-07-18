import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("data/tu_lanh_processed_vi.json")
OUTPUT_FILE = Path("data/tu_lanh_embedded.json")

MODEL_NAME = "intfloat/multilingual-e5-small"

BATCH_SIZE = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on: {device}")


model = SentenceTransformer(
    MODEL_NAME,
    device=device,
)

print("Model loaded!")


with INPUT_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} documents")


texts = [
    f"passage: {item['text']}"
    for item in data
]

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


with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=4,
    )

print(f"\nSaved {len(data)} documents")
print(f"Output: {OUTPUT_FILE}")