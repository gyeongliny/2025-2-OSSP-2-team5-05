import os
import os.path as osp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 이 파일 위치: project_root/graph_construction/prepare_notes/build_gatortron_embeddings.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_ROOT = osp.join(BASE_DIR, "data", "DATA_RAW", "root")
VOCAB_PATH = osp.join(EMB_ROOT, "vocab.txt")

MODEL_NAME = "UFNLP/gatortronS"


def main():
    # ✅ vocab 로드
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = [w.strip() for w in f.readlines() if w.strip()]

    print(f"[INFO] vocab size = {len(vocab):,}")

    # ✅ GatorTronS 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] loading model {MODEL_NAME} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    emb_dict = {}
    batch_size = 64

    with torch.no_grad():
        for i in tqdm(range(0, len(vocab), batch_size), desc="Embedding vocab with GatorTronS"):
            batch_words = vocab[i:i + batch_size]

            # 단어 리스트를 문장처럼 토크나이징
            inputs = tokenizer(
                batch_words,
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            # CLS 토큰 임베딩 사용
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")

            for w, vec in zip(batch_words, cls_emb):
                emb_dict[w] = vec

    # hidden size 자동 추출
    example_vec = next(iter(emb_dict.values()))
    emb_dim = int(example_vec.shape[-1])

    save_path = osp.join(EMB_ROOT, f"gatortron_{emb_dim}.npy")
    np.save(save_path, emb_dict)
    print(f"[INFO] saved GatorTron embeddings to: {save_path}")
    print(f"[INFO] embedding dim = {emb_dim}")


if __name__ == "__main__":
    main()
