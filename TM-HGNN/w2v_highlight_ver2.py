import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict, defaultdict
from matplotlib.ticker import MaxNLocator
import math
from matplotlib.patches import Ellipse 

# ==============================================================================
# [설정] 경로 및 저장소
# ==============================================================================
BASE_DIR = r"C:\Users\User\Desktop\OSS_2\TM-HGNN"
sys.path.append(BASE_DIR)

GRAPH_PATH = os.path.join(BASE_DIR, "2984_w2v_single_patient.pt")
CKPT_PATH  = os.path.join(BASE_DIR, "w2v_TM_HGNN.pth")
SAVE_DIR   = os.path.join(BASE_DIR, "3000_W2V_Fixed_Scale_Final") # 저장 폴더 변경
VOCAB_PATH = os.path.join(BASE_DIR, "data/DATA_RAW/root/vocab.txt")

os.makedirs(SAVE_DIR, exist_ok=True)

# PyG Safe Globals
from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage, NodeStorage, EdgeStorage])

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# [스타일 설정]
# ==============================================================================
WORD_COLOR_MAP = {
    # 2. 감염 (초록)
    "sepsis": "#32CD32",     
    "antibiotic": "#32CD32", 

    # 3. 혈류역학 (파랑)
    "hypotension": "#1E90FF",
    # "levophed": "#1E90FF",   
    # "bp": "#1E90FF",         
    # "cv": "#1E90FF",         
    "blood": "#1E90FF",      

    # 4. 호흡기 (핑크)
    # "vent": "#FF1493",
    "wean": "#FF1493",
    # "abg": "#FF1493",
    "resp": "#FF1493",       
}

TARGET_WORDS = list(WORD_COLOR_MAP.keys())

# ==============================================================================
# [모델 정의]
# ==============================================================================
from torch_geometric.nn import GCNConv
class TM_HGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):  
        super(TM_HGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_mask, batch=None):
        x0 = x 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x1 = x 
        idx_1 = torch.where(edge_mask==1)[0]
        if idx_1.numel() > 0:
            x = self.conv2(x, edge_index[:, idx_1])   
            x = x.relu()
        x2 = x 
        idx_2 = torch.where(edge_mask==2)[0]
        if idx_2.numel() > 0:
            x = self.conv3(x, edge_index[:, idx_2])        
        x3 = x 
        return (x0, x1, x2, x3)

# ==============================================================================
# [데이터 로드]
# ==============================================================================
def load_and_extract_embeddings():
    print(f"[INFO] Loading data from {GRAPH_PATH}...")
    data = torch.load(GRAPH_PATH, map_location=device)
    
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = f.read().split()

    print(f"[INFO] Loading weights from {CKPT_PATH}...")
    raw_state = torch.load(CKPT_PATH, map_location=device)
    state_dict = raw_state['state_dict'] if 'state_dict' in raw_state else raw_state

    new_state = OrderedDict()
    for k, v in state_dict.items():
        nk = k.replace("module.", "")
        new_state[nk] = v

    if "conv1.lin.weight" in new_state:
        weight_in_dim = new_state["conv1.lin.weight"].shape[1] 
        hidden_dim = new_state["conv1.lin.weight"].shape[0]    
    else:
        weight_in_dim = 104
        hidden_dim = 64

    # Slicing (260 -> 104)
    if data.x_n.size(1) > weight_in_dim:
        print(f"[ACTION] Slicing data input {data.x_n.size(1)} -> {weight_in_dim}")
        data.x_n = data.x_n[:, :weight_in_dim]

    model = TM_HGNN(num_features=weight_in_dim, hidden_channels=hidden_dim).to(device)
    
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[CHECK] Weights Loaded. Missing: {len(missing)}")
    
    model.eval()

    with torch.no_grad():
        (x0, x1, x2, x3) = model(data.x_n, data.edge_index_n, data.edge_index_mask)

    return (x0.cpu().numpy(), x1.cpu().numpy(), x2.cpu().numpy(), x3.cpu().numpy()), data, vocab

(x0_np, x1_np, x2_np, x3_np), data, vocab = load_and_extract_embeddings()

# 단어 인덱스 및 텍스트 매핑
node_types = data.x_n[:, 0].cpu().numpy().astype(int)
word_mask = (node_types == 0)
word_indices = data.x_n[:, 1].cpu().numpy().astype(int)
final_words = [vocab[idx] if 0 <= idx < len(vocab) else "unk" for idx in word_indices[word_mask]]

# ==============================================================================
# [시각화 함수] ClinicalBERT와 동일한 스타일 적용 (축 고정, 방사형, 강조)
# ==============================================================================
def run_pca(emb):
    emb = StandardScaler().fit_transform(emb)
    pca = PCA(n_components=2)
    return pca.fit_transform(emb)

def save_pca_single(emb, word_texts, title, filename_base):
    emb2d = run_pca(emb)
    
    # 텍스트 리스트
    word_texts = np.array(word_texts)

    # 1. 타겟 그룹 데이터 수집
    target_groups = defaultdict(list)
    for i, word in enumerate(word_texts):
        if word in TARGET_WORDS:
            x_t, y_t = emb2d[i]
            target_groups[word].append([x_t, y_t])

    # 2. 중심점 및 색상 정보 계산
    centroids = []
    for word, coords_list in target_groups.items():
        coords = np.array(coords_list)
        mean_x = np.mean(coords[:, 0])
        mean_y = np.mean(coords[:, 1])
        group_color = WORD_COLOR_MAP.get(word, 'yellow')
        
        centroids.append({
            'word': word, 'x': mean_x, 'y': mean_y, 
            'color': group_color 
        })

    # 3. 두 가지 모드로 저장
    modes = [("_label.png", True), ("_point.png", False)]

    for suffix, draw_text in modes:
        fig, ax = plt.subplots(figsize=(6, 6))

        # [A] 축 스케일 고정 (-50 ~ 50) - ClinicalBERT와 동일하게
        FIXED_LIMIT = 50
        ax.set_xlim(-FIXED_LIMIT, FIXED_LIMIT)
        ax.set_ylim(-FIXED_LIMIT, FIXED_LIMIT)

        # [B] 그룹별 점 찍기
        for c in centroids:
            ax.scatter([c['x']], [c['y']], 
                       s=200,               
                       alpha=0.6,           
                       color=c['color'],   
                       edgecolor='black',  
                       linewidth=2.5,      
                       zorder=10
                       )

        # [C] 텍스트 및 방사형 배치 (ClinicalBERT 로직과 동일)
        if draw_text:
            x_min, x_max = emb2d[:, 0].min(), emb2d[:, 0].max()
            y_min, y_max = emb2d[:, 1].min(), emb2d[:, 1].max()
            center_x, center_y = (x_min + x_max)/2, (y_min + y_max)/2
            data_width, data_height = (x_max - x_min)/2, (y_max - y_min)/2
            
            # 반경 설정
            radius_factor = 1.6
            radius_x = max(data_width * radius_factor, 15.0)
            radius_y = max(data_height * radius_factor, 15.0)

            # 각도 계산
            labels_info = []
            for c in centroids:
                angle = math.atan2(c['y'] - center_y, c['x'] - center_x)
                labels_info.append(c.copy())
                labels_info[-1]['angle'] = angle
            
            # 각도순 정렬
            labels_info.sort(key=lambda x: x['angle'])
            
            # 겹침 방지
            if len(labels_info) > 1:
                min_angle_diff = 2 * math.pi / (len(labels_info) * 1.2)
                for _ in range(3):
                    for i in range(len(labels_info)):
                        curr = labels_info[i]
                        prev = labels_info[i-1]
                        diff = curr['angle'] - prev['angle']
                        if diff < 0: diff += 2 * math.pi
                        if diff < min_angle_diff:
                            curr['angle'] = prev['angle'] + min_angle_diff
                            if curr['angle'] > math.pi: curr['angle'] -= 2*math.pi

            # 화살표 및 텍스트 그리기
            for info in labels_info:
                text_x = center_x + radius_x * math.cos(info['angle'])
                text_y = center_y + radius_y * math.sin(info['angle'])
                
                t = ax.annotate(
                    info['word'],
                    xy=(info['x'], info['y']), 
                    xytext=(text_x, text_y),   
                    arrowprops=dict(
                        arrowstyle="->", 
                        color="black", 
                        alpha=1.0,
                        lw=1.5,
                        connectionstyle="arc3,rad=0.1", 
                        shrinkB=8
                    ),
                    fontsize=15, weight='bold', color="black", 
                    ha='center', va='center', zorder=11
                )
                t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

            # [D] 의학적 인사이트 강조 (Annotation) - ClinicalBERT와 동일한 로직 적용
            # (a) Input Embedding: Sepsis(감염) - Hypotension(혈류)
            if "Input" in title:
                p1 = next((c for c in centroids if c['word'] == 'sepsis'), None)
                p2 = next((c for c in centroids if c['word'] == 'hypotension'), None)
                if p1 and p2:
                    mid_x, mid_y = (p1['x'] + p2['x'])/2, (p1['y'] + p2['y'])/2
                    width = abs(p1['x'] - p2['x']) + 15
                    height = abs(p1['y'] - p2['y']) + 15
                    ell = Ellipse((mid_x, mid_y), width=width, height=height, 
                                  angle=0, color='red', alpha=0.1, zorder=0)
                    ax.add_patch(ell)
                    ax.annotate("Clinical Relation\n(Septic Shock)", 
                                xy=(mid_x, mid_y), xytext=(mid_x + 10, mid_y + 10),
                                arrowprops=dict(facecolor='black', shrink=0.05, lw=1),
                                fontsize=14, color='darkred', weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
                                zorder=30)
            
            # (c) Note Layer: Respiratory Cluster
            if "Note" in title:
                resp_words = ['vent', 'wean', 'resp']
                resp_centroids = [c for c in centroids if c['word'] in resp_words]
                if resp_centroids:
                    r_x = np.mean([c['x'] for c in resp_centroids])
                    r_y = np.mean([c['y'] for c in resp_centroids])
                    ax.annotate("Distinct\nRespiratory Cluster", 
                                xy=(r_x, r_y), xytext=(r_x + 20, r_y - 20),
                                arrowprops=dict(facecolor='black', shrink=0.05, lw=1),
                                fontsize=14, color='#C71585', weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#FF1493", alpha=0.8),
                                zorder=30)

        # 공통 스타일
        ax.set_title(title, fontsize=30, loc='left', pad=10)
        ax.set_xlabel("PC1", fontsize=25, labelpad=20)
        ax.set_ylabel("PC2", fontsize=25, labelpad=20)
        ax.grid(True, alpha=0.2, linewidth=2)
        for spine in ax.spines.values(): spine.set_linewidth(3)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.tick_params(axis='both', which='major', width=3, length=10)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, filename_base + suffix)
        plt.savefig(save_path, dpi=350)
        plt.close()
        print(f"[SAVE] {title} -> {save_path}")

# ==============================================================================
# [범례 저장]
# ==============================================================================
def save_legend_only():
    fig, ax = plt.subplots(figsize=(6, 6))
    h, l = [], []

    h.append(ax.scatter([],[],s=0)); l.append("Semantic Topic")
    
    topic_map = {
        "Infection": "#32CD32", 
        "Hemodynamics": "#1E90FF",
        "Respiratory": "#FF1493",
    }
    
    for t, c in topic_map.items():
        h.append(ax.scatter([],[],s=200,color=c,edgecolor='black',linewidth=2.0))
        l.append(t)
    
    leg = ax.legend(h, l, fontsize=14, loc='center', frameon=True, labelspacing=1.2, edgecolor='black', framealpha=1.0)
    leg.get_frame().set_linewidth(3.0)
    
    for t in leg.get_texts():
        if t.get_text() == "Semantic Topic":
            t.set_weight("bold")
            t.set_fontsize(16)
            t.set_ha('center')
            t.set_position((45,0))

    ax.axis('off')
    fig.savefig(os.path.join(SAVE_DIR, "00_legend.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# [실행]
# ==============================================================================
# x0: Word2Vec Input (4개 구조정보 제외)
x0_words = x0_np[word_mask]
save_pca_single(x0_words[:, 4:], final_words, "(a) Input Embedding", "01_input") 

# x1, x2, x3: 모델 출력
save_pca_single(x1_np[word_mask], final_words, "(b) Initialization Layer", "02_init")
save_pca_single(x2_np[word_mask], final_words, "(c) Note Layer", "03_note")
save_pca_single(x3_np[word_mask], final_words, "(d) Taxonomy Layer", "04_taxonomy")

save_legend_only()
print(f"\n[DONE] All images saved to {SAVE_DIR}")