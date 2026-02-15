import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from matplotlib.ticker import MaxNLocator
from transformers import AutoTokenizer

# ===========================================================
# 0) Path
# ===========================================================
BASE_DIR = r"C:\Users\User\Desktop\OSS_2\TM-HGNN"
sys.path.append(BASE_DIR)

GRAPH_PATH = os.path.join(BASE_DIR, "2984_clinical_single_patient.pt")
CKPT_PATH  = os.path.join(BASE_DIR, "cinical_TM_HGNN.pth")
SAVE_DIR   = os.path.join(BASE_DIR, "2984_pca_clinicalbert")

os.makedirs(SAVE_DIR, exist_ok=True)

from tmhgnn.net import TM_HGNN
from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage

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

# ===========================================================
# 2) Load Model
# ===========================================================
raw_state = torch.load(CKPT_PATH, map_location="cpu")
hidden_channels = 128  # 에러 메시지에서 확인된 정답 차원수
in_channels = data.x_n.size(1)

model = TM_HGNN(
    num_features=in_channels,
    hidden_channels=hidden_channels,
).to(device)

new_state = OrderedDict()
for k, v in raw_state.items():
    nk = k[7:] if k.startswith("module.") else k
    new_state[nk] = v

model.load_state_dict(new_state, strict=False)
model.eval()

# ===========================================================
# 3) Extract Embeddings
# ===========================================================
@torch.no_grad()
def extract_embeddings(model, data):
    logits, (x0, x1, x2, x3) = model(
        x=data.x_n,
        edge_index=data.edge_index_n,
        edge_mask=data.edge_index_mask,
        batch=data.batch_n,
        return_all=True
    )
    return (
        x0.cpu().numpy(),
        x1.cpu().numpy(),
        x2.cpu().numpy(),
        x3.cpu().numpy(),
    )

x0_np, x1_np, x2_np, x3_np = extract_embeddings(model, data)

# ===========================================================
# 4) Build category propagation to word nodes
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

# ClinicalBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

TARGET_WORDS = ["rhythm", "fibrillation", "benadryl"]
target_word_ids = {w: tokenizer.convert_tokens_to_ids(w) for w in TARGET_WORDS}

# word 노드들의 global index & word_id
global_word_idx = np.where(word_mask)[0]
word_ids_global = data.x_n[:, 1].cpu().numpy().astype(int)  # x_n[:,1]에 word_id 저장돼 있다고 가정

# ===========================================================
# PCA helper
# ===========================================================
def run_pca(emb):
    emb = StandardScaler().fit_transform(emb)
    pca = PCA(n_components=2)
    return pca.fit_transform(emb)

# ===========================================================
# (NEW) Eigenvalue Analysis Function
# ===========================================================
def analyze_pca_eigenvalues(embeddings, layer_name="Layer"):
    if hasattr(embeddings, "detach"):
        X = embeddings.detach().cpu().numpy()
    else:
        X = embeddings

    pca_full = PCA(n_components=min(X.shape[0], X.shape[1]))
    pca_full.fit(X)

    eigenvalues = pca_full.explained_variance_
    ratio = pca_full.explained_variance_ratio_

    print(f"\n===== [Eigenvalues] {layer_name} =====")
    print("Top 10 eigenvalues:", eigenvalues[:10])
    print("Top 10 variance ratios (%):", ratio[:10] * 100)

    # Scree plot
    plt.figure(figsize=(6,4))
    plt.plot(eigenvalues, marker='o')
    plt.title(f"Scree Plot - {layer_name}")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.grid(True, alpha=0.4)

    save_path = os.path.join(SAVE_DIR, f"{layer_name}_eigen.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SAVE] Eigenvalue plot ({layer_name}) → {save_path}")

    return eigenvalues, ratio

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

    # ====== (추가) 특정 단어 하이라이트 ======
    for w, wid in target_word_ids.items():
        # 이 word_id를 가진 global index들 중 첫 번째 사용
        candidate = [gi for gi in global_word_idx if word_ids_global[gi] == wid]
        if not candidate:
            continue  # 해당 episode에 그 단어가 없으면 skip

        g_idx = candidate[0]
        # global index → word-only 로컬 인덱스
        local_idx = np.where(global_word_idx == g_idx)[0][0]
        x_t, y_t = word_emb[local_idx]

        ax.scatter([x_t], [y_t],
                   s=220, color="yellow",
                   edgecolor="black", linewidth=1.8,
                   zorder=5)
        ax.text(x_t + 0.4, y_t + 0.4, w,
                fontsize=14, weight="bold")
        
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
# legend(범례) 저장 함수
# ===========================================================
def save_legend_only(filename="legend_only.png"):
    fig, ax = plt.subplots(figsize=(4, 4))

    # 더미 포인트 만들어서 legend만 생성
    handles = []
    labels = []

    for cname in CATEGORY_LIST:
        h = ax.scatter([], [], 
                       color=COLOR_MAP[cname], 
                       s=200, 
                       edgecolor='white', 
                       linewidth=0.5)
        handles.append(h)
        labels.append(cname)

    legend = ax.legend(handles, labels, 
                       fontsize=15,
                       markerscale=1.8,
                       frameon=True,
                       borderpad=1,
                       labelspacing=1)

    ax.axis('off')

    save_path = os.path.join(SAVE_DIR, filename)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[SAVE] Legend only → {save_path}")

# ===========================================================
# (NEW) Eigenvalue Analysis for each layer
# ===========================================================
analyze_pca_eigenvalues(x0_np[:,4:], "Input")
analyze_pca_eigenvalues(x1_np, "Init")
analyze_pca_eigenvalues(x2_np, "Note")
analyze_pca_eigenvalues(x3_np, "Taxonomy")

# ===========================================================
# Run all layers
# ===========================================================
save_pca_single(x0_np[:,4:], "(a) Input Embedding", "01_input.png")
save_pca_single(x1_np, "(b) Initialization Layer", "02_init.png")
save_pca_single(x2_np, "(c) Note Layer", "03_note.png")
save_pca_single(x3_np, "(d) Taxonomy Layer", "04_taxonomy.png")

# --- 범례만 따로 저장 ---
save_legend_only("00_legend_tm.png")

print("[DONE] ClinicalBERT PCA visualization complete.")