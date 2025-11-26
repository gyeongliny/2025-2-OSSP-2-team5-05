import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_RAW = osp.join(BASE_DIR, "data", "DATA_RAW")


def collect_notes(raw_path: str, task: str):
    """
    train_note + test_note 전체에서 'fixed TEXT' 컬럼만 수집
    """
    all_texts = []
    index_rows = []

    for split in ["train", "test"]:
        partition = split + "_note"
        note_dir = osp.join(raw_path, task, partition)

        if not osp.isdir(note_dir):
            print(f"[WARN] note_dir not found: {note_dir}")
            continue

        patients = [
            f for f in os.listdir(note_dir)
            if "episode" in f and f.endswith(".csv")
        ]

        for patient in tqdm(patients, desc=f"[{split}] Loading notes"):
            fpath = osp.join(note_dir, patient)
            df = pd.read_csv(fpath, sep="\t", header=0, engine="python")

            col = "fixed TEXT" if "fixed TEXT" in df.columns else "TEXT"

            for row_idx, text in enumerate(df[col].astype(str).tolist()):
                all_texts.append(text)
                index_rows.append({
                    "split": split,
                    "file": patient,
                    "row_idx": row_idx,
                })

    return all_texts, pd.DataFrame(index_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, default=DEFAULT_DATA_RAW)
    parser.add_argument("--task", type=str, default="in-hospital-mortality")
    parser.add_argument("--hf_model_name", type=str, default="UFNLP/gatortronS")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    emb_root = osp.join(args.raw_path, "root")
    os.makedirs(emb_root, exist_ok=True)

    print("[STEP 1] Collecting fixed-text notes...")
    texts, index_df = collect_notes(args.raw_path, args.task)
    print(f"[INFO] Total notes = {len(texts):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[STEP 2] Loading model: {args.hf_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    model = AutoModel.from_pretrained(args.hf_model_name)
    model.to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    print(f"[INFO] hidden_size = {hidden_size}")

    print("[STEP 3] Embedding notes...")
    all_embs = []
    bs = args.batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), bs), desc="Embedding notes"):
            batch = texts[i:i + bs]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            cls_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
            all_embs.append(cls_vec)

    emb_mat = np.vstack(all_embs)
    print(f"[INFO] emb_mat shape = {emb_mat.shape}")

    emb_path = osp.join(emb_root, f"gatortron_notes_{hidden_size}.npy")
    idx_path = osp.join(emb_root, "gatortron_notes_index.csv")

    np.save(emb_path, emb_mat)
    index_df.to_csv(idx_path, index=False)

    print(f"[STEP 4] Saved embeddings → {emb_path}")
    print(f"[STEP 4] Saved index → {idx_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
