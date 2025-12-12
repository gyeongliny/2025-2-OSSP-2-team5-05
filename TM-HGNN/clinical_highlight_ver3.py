import pandas as pd
import networkx as nx
import numpy as np
from scipy import sparse
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator
from torch_geometric.data import Data
import os
from collections import OrderedDict, defaultdict
import matplotlib.patheffects as PathEffects
import math
from matplotlib.patches import Ellipse 

# ==============================================================================
# [설정] 경로 및 저장소
# ==============================================================================
BASE_DIR = os.getcwd() 
SAVE_DIR = os.path.join(BASE_DIR, "2984_highlight_fixed_scale_final") # 저장 폴더명
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_CSV = r'data/DATA_PRE/in-hospital-mortality/train_hyper/2984_episode5637.csv'
WEIGHTS_PATH = r'clinical_TM_HGNN.pth'
BERT_EMB_PATH = r'data/DATA_RAW/root/clinicalbert_768.npy'
VOCAB_PATH = r'data/DATA_RAW/root/vocab.txt'

from tmhgnn.net import TM_HGNN 

# ==============================================================================
# [스타일 설정]
# ==============================================================================
CATEGORY_LIST = ['Radiology', 'Nursing', 'Nursing/other', 'ECG', 'Echo', 'Physician']

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
# [함수] 데이터 처리 및 모델 로드 (기존 동일)
# ==============================================================================
def load_vocab(path):
    try:
        with open(path, 'r', encoding='utf-8') as f: return f.read().split()
    except: return ["dummy"] * 1000

def load_bert_embeddings(path):
    try:
        emb_dict = np.load(path, allow_pickle=True).item()
        return lambda w: emb_dict.get(w, np.zeros(768, dtype=np.float32)).astype(np.float32)
    except: return lambda w: np.random.randn(768).astype(np.float32)

def graph_to_sparse_tensor(G):
    G_int = nx.convert_node_labels_to_integers(G)
    A = nx.to_numpy_array(G_int, weight='edge_type')
    sparse_mx = sparse.csr_matrix(A).tocoo()
    edge_index = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    x, batch_n, batch_t = [], [], []
    for i in range(len(G_int)):
        node = G_int.nodes[i]
        x.append(node.get('node_emb', np.zeros(772)))
        batch_n.append(node.get('note_id', -1))
        batch_t.append(node.get('cat_id', -1))
    return (torch.tensor(np.array(x), dtype=torch.float32), edge_index, 
            torch.from_numpy(sparse_mx.data).long(), 
            torch.tensor(np.array(batch_n), dtype=torch.long),
            torch.tensor(np.array(batch_t), dtype=torch.long))

def process_single_csv(csv_path, vocab, get_vec_func):
    try:
        df = pd.read_csv(csv_path, sep='\t')
        if 'WORD' not in df.columns and 'word' in df.columns: df.rename(columns={'word':'WORD'}, inplace=True)
    except: return None, []
    df = df.dropna()
    df['CATEGORY'] = df['CATEGORY'].astype(str).str.strip()
    df['WORD'] = df['WORD'].astype(str)
    df['note_NM'] = "n_" + df['note_id'].astype(str)
    cat_to_id = {cat: i for i, cat in enumerate(CATEGORY_LIST)}
    df = df[df['CATEGORY'].isin(CATEGORY_LIST)]
    if len(df) == 0: return None, []
    G_list = []
    for _, n_df in df.groupby('note_NM'):
        n_df = n_df.copy()
        n_df['edge_type'] = 1 
        G_n = nx.from_pandas_edgelist(n_df, 'WORD', 'note_NM', 'edge_type')
        cat_id = cat_to_id.get(n_df['CATEGORY'].values[0].strip(), -1)
        note_id = int(str(n_df['note_id'].values[0]).replace('n_',''))
        tax_node = f"t_{cat_id}"
        nodes_to_add_edge = [node for node in G_n.nodes() if not str(node).startswith('n_')]
        for word_node in nodes_to_add_edge:
            G_n.add_edge(word_node, tax_node, edge_type=2)
        for node in G_n.nodes():
            emb = np.zeros(4 + 768, dtype=np.float32)
            if str(node).startswith('n_'):
                emb[0], emb[2], emb[3] = 1, note_id, cat_id
            elif str(node).startswith('t_'):
                emb[0], emb[2], emb[3] = 2, -1, cat_id
            else:
                emb[0], emb[2], emb[3] = 0, note_id, cat_id
                emb[1] = vocab.index(node) if node in vocab else -1
                emb[4:] = get_vec_func(node)
            G_n.nodes[node]['node_emb'] = emb
            G_n.nodes[node]['note_id'] = emb[2]
            G_n.nodes[node]['cat_id'] = emb[3]
        G_list.append(G_n)
    if not G_list: return None, []
    G_final = nx.disjoint_union_all(G_list)
    x, edge_index, edge_mask, batch_n, batch_t = graph_to_sparse_tensor(G_final)
    data = Data(x=x, edge_index=edge_index, edge_mask=edge_mask, 
                batch=torch.zeros(x.size(0), dtype=torch.long),
                batch_t=batch_t, batch_n=batch_n)
    return data

def load_weights_compat(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if 'convs.0' in name: name = name.replace('convs.0', 'conv1')
        elif 'convs.1' in name: name = name.replace('convs.1', 'conv2')
        elif 'convs.2' in name: name = name.replace('convs.2', 'conv3')
        if 'bns.0' in name: name = name.replace('bns.0.module', 'bn1').replace('bns.0', 'bn1')
        elif 'bns.1' in name: name = name.replace('bns.1.module', 'bn2').replace('bns.1', 'bn2')
        elif 'bns.2' in name: name = name.replace('bns.2.module', 'bn3').replace('bns.2', 'bn3')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

# ==============================================================================
# [시각화 함수] 원래 스타일 복원 (fontsize, linewidth 등)
# ==============================================================================
def run_pca(emb):
    emb = StandardScaler().fit_transform(emb)
    pca = PCA(n_components=2)
    return pca.fit_transform(emb)

def save_pca_single(emb, word_texts, title, filename_base):
    emb2d = run_pca(emb)
    
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

    modes = [("_label.png", True), ("_point.png", False)]

    for suffix, draw_text in modes:
        # [STYLE] 원래 코드의 figsize (6,6) 유지 (축 고정을 위해 약간 여유가 필요하면 키울 수도 있음)
        fig, ax = plt.subplots(figsize=(6, 6))

        # [CHANGE] 축 스케일 고정 (User Request: -50 ~ 50)
        FIXED_LIMIT = 50
        ax.set_xlim(-FIXED_LIMIT, FIXED_LIMIT)
        ax.set_ylim(-FIXED_LIMIT, FIXED_LIMIT)

        # [STYLE] 원래 코드의 점 스타일 유지
        for c in centroids:
            ax.scatter([c['x']], [c['y']], 
                       s=200,                # 원래 크기
                       alpha=0.6,            # 원래 투명도
                       color=c['color'],   
                       edgecolor='black',  
                       linewidth=2.5,        # 원래 테두리 두께
                       zorder=10
                       )

        if draw_text:
            # [LOGIC] 원래 코드의 방사형(Radial) 배치 로직 복원
            x_min, x_max = emb2d[:, 0].min(), emb2d[:, 0].max()
            y_min, y_max = emb2d[:, 1].min(), emb2d[:, 1].max()
            
            # (축은 고정되었지만, 라벨 위치 계산을 위해 데이터 중심은 유지)
            center_x, center_y = (x_min + x_max)/2, (y_min + y_max)/2
            data_width, data_height = (x_max - x_min)/2, (y_max - y_min)/2
            
            radius_factor = 1.6
            # 데이터가 너무 모여있을 경우를 대비해 최소 반경 보장 (축이 50이므로 적절히 조정)
            radius_x = max(data_width * radius_factor, 15.0)
            radius_y = max(data_height * radius_factor, 15.0)

            labels_info = []
            for c in centroids:
                angle = math.atan2(c['y'] - center_y, c['x'] - center_x)
                labels_info.append(c.copy())
                labels_info[-1]['angle'] = angle
            
            labels_info.sort(key=lambda x: x['angle'])
            
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

            for info in labels_info:
                text_x = center_x + radius_x * math.cos(info['angle'])
                text_y = center_y + radius_y * math.sin(info['angle'])
                
                # [STYLE] 원래 폰트 크기(15) 및 화살표 스타일 복원
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

            # [ADD] 의학적 인사이트 강조 (Annotation) 추가
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
                    # 텍스트가 라벨과 겹치지 않도록 위치 미세 조정
                    ax.annotate("Clinical Relation\n(Septic Shock)", 
                                xy=(mid_x, mid_y), xytext=(mid_x + 10, mid_y + 10),
                                arrowprops=dict(facecolor='black', shrink=0.05, lw=1),
                                fontsize=14, color='darkred', weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
                                zorder=30)

        # [STYLE] 원래 코드의 타이틀, 축 라벨, 스파인 스타일 복원
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
        plt.savefig(save_path, dpi=350) # DPI 350 유지
        plt.close()
        print(f"[SAVE] {title} -> {save_path}")

def save_legend_only(filename="00_legend.png"):
    fig, ax = plt.subplots(figsize=(6, 7))
    handles, labels = [], []

    handles.append(ax.scatter([], [], s=0)); labels.append("")
    handles.append(ax.scatter([], [], s=0)); labels.append("Semantic Topic")

    topic_map = {
        "Infection": "#32CD32", 
        "Hemodynamics": "#1E90FF",
        "Respiratory": "#FF1493",
    }

    for topic, color in topic_map.items():
        h = ax.scatter([], [], 
                       s=200, color=color, edgecolor='black', linewidth=2.5, alpha=0.7)
        handles.append(h)
        labels.append(topic)

    legend = ax.legend(handles, labels, 
              fontsize=15, markerscale=1.2, frameon=True, borderpad=1.5, labelspacing=1.2,
              loc='center', framealpha=1.0, edgecolor='black')
    legend.get_frame().set_linewidth(3.0)
    
    title_shift = 40 
    for text in legend.get_texts():
        txt = text.get_text()
        if txt in ["Semantic Topic"]:
            text.set_weight("bold")
            text.set_fontsize(18)
            text.set_color("black")
            text.set_ha('center') 
            text.set_position((title_shift, 0))

    ax.axis('off')
    fig.savefig(os.path.join(SAVE_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# [메인 실행]
# ==============================================================================
def main():
    device = torch.device('cpu')
    vocab = load_vocab(VOCAB_PATH)
    get_vec = load_bert_embeddings(BERT_EMB_PATH)
    print("[Info] Processing Data...")
    data = process_single_csv(TARGET_CSV, vocab, get_vec)
    if data is None: return
    print("[Info] Initializing Model...")
    model = TM_HGNN(num_features=772, hidden_channels=128, proj_dim=256, dropout=0.3).to(device)
    if os.path.exists(WEIGHTS_PATH):
        load_weights_compat(model, WEIGHTS_PATH, device)
    else:
        print("[Error] Weights file not found!")
        return
    model.eval()
    print("[Info] Extracting Embeddings...")
    with torch.no_grad():
        logits, (x0, x1, x2, x3) = model(data.x, data.edge_index, data.edge_mask, data.batch, return_all=True)
    
    word_mask = (data.x[:, 0] == 0).numpy()
    word_indices = data.x[:, 1].long().numpy()[word_mask]
    word_texts = [vocab[idx] if 0 <= idx < len(vocab) else "unk" for idx in word_indices]

    save_pca_single(x0.numpy()[word_mask][:, 4:], word_texts, "(a) Input Embedding", "01_input")
    save_pca_single(x1.numpy()[word_mask], word_texts, "(b) Initialization Layer", "02_init")
    save_pca_single(x2.numpy()[word_mask], word_texts, "(c) Note Layer", "03_note")
    save_pca_single(x3.numpy()[word_mask], word_texts, "(d) Taxonomy Layer", "04_taxonomy")
    
    save_legend_only()
    print(f"\n[DONE] All images saved in: {SAVE_DIR}")

if __name__ == '__main__':
    main()