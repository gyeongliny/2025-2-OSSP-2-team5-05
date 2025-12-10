import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from matplotlib.ticker import MaxNLocator

# ===========================================================
# 0) Path
# ===========================================================
BASE_DIR = r"C:\Users\User\Desktop\OSS_2\TM-HGNN"
sys.path.append(BASE_DIR)

GRAPH_PATH = os.path.join(BASE_DIR, "8498_w2v_single_patient.pt")
CKPT_PATH  = os.path.join(BASE_DIR, "w2v_TM_HGNN.pth")
SAVE_DIR   = os.path.join(BASE_DIR, "8498_pca_w2v_layers")

os.makedirs(SAVE_DIR, exist_ok=True)

from tmhgnn.net import TM_HGNN
from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage

# allow loading PyG Data
torch.serialization.add_safe_globals([
    Data, DataEdgeAttr, DataTensorAttr,
    GlobalStorage, NodeStorage, EdgeStorage
])

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================================================
# 1) Load Graph
# ===========================================================
print("[INFO] Loading single_patient.pt...")
data = torch.load(GRAPH_PATH, map_location="cpu")
data = data.to(device)

print("Loaded graph:")
print(" - x_n:", data.x_n.shape)
print(" - y_n:", data.y_n.shape)
print(" - edge_index:", data.edge_index_n.shape)

# load vocab
vocab_path = os.path.join(BASE_DIR, "data/DATA_RAW/root/vocab.txt")
with open(vocab_path, "r", encoding="utf-8") as f:
    dictionary = f.read().split()

# ===========================================================
# 2) Load Model
# ===========================================================
raw_state = torch.load(CKPT_PATH, map_location="cpu")
hidden_channels = raw_state["conv1.bias"].shape[0]
in_channels = data.x_n.size(1)

model = TM_HGNN(
    num_features=in_channels,
    hidden_channels=hidden_channels,
).to(device)

new_state = OrderedDict()
for k, v in raw_state.items():
    nk = k[7:] if k.startswith("module.") else k
    new_state[nk] = v

model.load_state_dict(new_state, strict=True)
model.eval()

# ===========================================================
# 3) Extract Layer Embeddings (NO return_all)
# ===========================================================
@torch.no_grad()
def extract_word2vec_layers(data):
    x0 = data.x_n                     # 104차원 전체 입력

    # Layer 1
    x1 = model.conv1(x0, data.edge_index_n)
    x1 = x1.relu()

    # Layer 2
    idx1 = torch.where(data.edge_index_mask == 1)[0]
    x2 = model.conv2(x1, data.edge_index_n[:, idx1])
    x2 = x2.relu()

    # Layer 3
    idx2 = torch.where(data.edge_index_mask == 2)[0]
    x3 = model.conv3(x2, data.edge_index_n[:, idx2])

    return (
        x0.cpu().numpy(),
        x1.cpu().numpy(),
        x2.cpu().numpy(),
        x3.cpu().numpy()
    )


x0_np, x1_np, x2_np, x3_np = extract_word2vec_layers(data)

# ===========================================================
# 4) Category Propagation to word nodes
# ===========================================================
node_types = data.x_n[:, 0].cpu().numpy().astype(int)
word_mask = (node_types == 0)
note_mask = (node_types == 1)

cat = np.full(len(node_types), -1)
note_cat = data.y_n[:, 0].cpu().numpy()
cat[note_mask] = note_cat

edge_src, edge_dst = data.edge_index_n.cpu().numpy()

for s, d in zip(edge_src, edge_dst):
    if note_mask[s] and word_mask[d]:
        cat[d] = cat[s]
    if note_mask[d] and word_mask[s]:
        cat[s] = cat[d]

CATEGORY_LIST = ['Radiology', 'Nursing', 'Nursing/other', 'ECG', 'Echo', 'Physician']
COLOR_MAP = {
    'Radiology': "#d62728",
    'Nursing': "#1f77b4",
    'Nursing/other': "#2ca02c",
    'ECG': "#ff7f0e",
    'Echo': "#9467bd",
    'Physician': "#8c564b",
}

TARGET_WORDS = ["rhythm", "fibrillation", "benadryl"]
target_word_ids = {w: dictionary.index(w) for w in TARGET_WORDS if w in dictionary}

global_word_idx = np.where(word_mask)[0]
word_ids_global = data.x_n[:, 1].cpu().numpy().astype(int)

# ===========================================================
# [Modified] 4.5) Target Word Debugging & Preparation
# ===========================================================

# 1. 대소문자 통일 (무조건 소문자로 검색)
#    (리스트에 있는 단어들을 소문자로 변환하여 검색)
raw_targets = ["benadryl", "fibrillation", "rhythm"] 
TARGET_WORDS = [w.lower() for w in raw_targets]

# 2. Dictionary 매칭 확인
target_word_ids = {}
print("\n[DEBUG] Target Word Search in Dictionary:")
for w in TARGET_WORDS:
    if w in dictionary:
        wid = dictionary.index(w)
        target_word_ids[w] = wid
        print(f"  - '{w}' found in vocab (ID: {wid})")
    else:
        print(f"  - [WARNING] '{w}' is NOT in vocab.txt (Check spelling/casing)")

# 3. Graph 내 존재 여부 확인 (가장 중요!)
print("\n[DEBUG] Target Word Existence in This Patient Graph:")
global_word_idx = np.where(word_mask)[0]
word_ids_global = data.x_n[:, 1].cpu().numpy().astype(int) # 노드의 단어 ID 컬럼

final_target_map = {} # 실제 그릴 수 있는 단어만 저장

for w, wid in target_word_ids.items():
    # 그래프 내의 모든 단어 ID 중, 우리가 찾는 wid가 있는지 확인
    matching_indices = np.where(word_ids_global[global_word_idx] == wid)[0]
    
    if len(matching_indices) > 0:
        print(f"  - [OK] '{w}' exists in this patient graph! (Count: {len(matching_indices)})")
        final_target_map[w] = wid
    else:
        print(f"  - [MISSING] '{w}' (ID: {wid}) is NOT in patient 8498's graph.")

# -----------------------------------------------------------
# [Tip] 만약 위에서 [MISSING]이 뜬다면, 환자가 실제로 가진 단어로 바꿔보세요.
# 아래 코드는 이 환자가 가장 많이 가진 단어 Top 5를 보여줍니다.
print("\n[INFO] Top 5 most frequent words in this patient:")
from collections import Counter
existing_ids = word_ids_global[global_word_idx]
top_ids = Counter(existing_ids).most_common(5)
for tid, count in top_ids:
    t_word = dictionary[tid]
    print(f"  - Word: '{t_word}', Count: {count}")
# -----------------------------------------------------------

# ===========================================================
# Plot function (수정된 highlight 로직 적용)
# ===========================================================
def save_pca_single(emb, title, filename):
    emb2d = run_pca(emb)

    word_emb = emb2d[word_mask]
    word_cat = cat[word_mask]
    word_names = np.array([CATEGORY_LIST[c] for c in word_cat])

    fig, ax = plt.subplots(figsize=(8, 8)) # 크기 약간 키움

    # 1. 일반 점 찍기
    for cname in np.unique(word_names):
        mask = (word_names == cname)
        ax.scatter(
            word_emb[mask, 0], 
            word_emb[mask, 1],
            s=100, 
            alpha=0.6,
            color=COLOR_MAP.get(cname, 'gray'), # 안전하게 get 사용
            edgecolor='white', 
            linewidth=0.3,
            label=cname # 범례 추가
        )

    # 2. Highlight 찍기 (final_target_map 사용)
    # word_ids_global은 전체 노드 기준이므로, word_emb(워드 마스크된 것)와 인덱스 맞추기 주의
    
    # word_mask가 True인 인덱스들만 뽑은게 global_word_idx
    # word_ids_global[global_word_idx] 가 word_emb 의 각 행에 대응되는 단어 ID임
    
    current_patient_word_ids = word_ids_global[global_word_idx]

    for w, wid in final_target_map.items():
        # 이 환자의 단어 리스트에서 해당 ID를 가진 인덱스 찾기
        local_indices = np.where(current_patient_word_ids == wid)[0]
        
        if len(local_indices) > 0:
            # 하나만 찍거나(첫번째), 전부 찍거나. 여기선 첫번째 등장만 표시
            idx = local_indices[0] 
            x_t, y_t = word_emb[idx]

            ax.scatter([x_t], [y_t], 
                       s=300, 
                       marker='*', # 별표로 강조
                       color="yellow",
                       edgecolor="black", 
                       linewidth=1.5, 
                       zorder=10)
            
            ax.text(x_t + 0.2, y_t + 0.2, w, 
                    fontsize=14, weight="bold", color='black', zorder=11)

    ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 범례 표시 (중복 제거)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[SAVE] {title} → {save_path}")
    
# ===========================================================
# PCA helper
# ===========================================================
def run_pca(emb):
    emb = StandardScaler().fit_transform(emb)
    pca = PCA(n_components=2)
    return pca.fit_transform(emb)

# ===========================================================
# Plot function
# ===========================================================
def save_pca_single(emb, title, filename):
    emb2d = run_pca(emb)

    word_emb = emb2d[word_mask]
    word_cat = cat[word_mask]
    word_names = np.array([CATEGORY_LIST[c] for c in word_cat])

    fig, ax = plt.subplots(figsize=(6, 6))

    for cname in np.unique(word_names):
        mask = (word_names == cname)
        ax.scatter(
            word_emb[mask, 0], 
            word_emb[mask, 1],
            s=100, 
            alpha=0.6,
            color=COLOR_MAP[cname],
            edgecolor='white', 
            linewidth=0.3
        )

    # highlight keywords
    for w, wid in target_word_ids.items():
        candidate = [gi for gi in global_word_idx if word_ids_global[gi] == wid]
        if not candidate:
            continue
        g_idx = candidate[0]
        local_idx = np.where(global_word_idx == g_idx)[0][0]
        x_t, y_t = word_emb[local_idx]

        ax.scatter([x_t], [y_t], 
                   s=220, 
                   color="yellow",
                   edgecolor="black", 
                   linewidth=1.8, 
                   zorder=5)
        ax.text(x_t + 0.3, y_t + 0.3, w, 
                fontsize=12, weight="bold")

    ax.set_title(title, fontsize=30, loc='left', pad=10)
    ax.set_xlabel("PC1", fontsize=25, labelpad=20)
    ax.set_ylabel("PC2", fontsize=25, labelpad=20)
    ax.grid(True, alpha=0.2, linewidth=2)

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.tick_params(axis='both', which='major', width=3, length=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=350)
    plt.close()

    print(f"[SAVE] {title} → {save_path}")

# ===========================================================
# Run PCA for all layers
# ===========================================================
save_pca_single(x0_np, "(a) Input Embedding", "01_input.png")
save_pca_single(x1_np, "(b) Initialization Layer", "02_conv1.png")
save_pca_single(x2_np, "(c) Note Layer", "03_conv2.png")
save_pca_single(x3_np, "(d) Taxonomy Layer", "04_conv3.png")

print("[DONE] Word2Vec PCA visualization complete.")
