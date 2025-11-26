import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import os.path as osp
import glob

import pandas as pd
import networkx as nx
import numpy as np
from scipy import sparse
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from gensim.models import Word2Vec

pd.set_option('display.max_columns', None)


def graph_to_torch_sparse_tensor(G_true, node_attr=None):
    """
    networkx 그래프 → PyTorch Geometric에서 사용하는
    edge_index / edge_attr / x / batch_t / batch_n 형태로 변환
    """
    G = nx.convert_node_labels_to_integers(G_true)
    A_G = nx.to_numpy_array(G, weight='edge_type', dtype=float)

    # scipy sparse → torch sparse style (edge_index + edge_attr)
    sparse_mx = sparse.csr_matrix(A_G).tocoo()
    edge_index = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col))
    ).to(torch.long)
    edge_attrs = torch.from_numpy(sparse_mx.data).to(torch.float32)

    x = []
    batch_n = []
    batch_t = []
    for node in range(len(G)):
        x.append(G.nodes[node]['node_emb'])
        if node_attr is not None:
            for attr in node_attr:
                if attr == 'note_id':
                    batch_n.append(G.nodes[node][attr])
                elif attr == 'cat_id':
                    batch_t.append(G.nodes[node][attr])

    x = torch.from_numpy(np.array(x)).to(torch.float32)
    batch_n = torch.from_numpy(np.array(batch_n)).to(torch.long)
    batch_t = torch.from_numpy(np.array(batch_t)).to(torch.long)

    return edge_index, edge_attrs, x, batch_t, batch_n


# === 경로 설정: 프로젝트 루트 기준으로 DATA_RAW/root ===
# 이 파일 위치: project_root/graph_construction/prepare_notes/ConstructDatasetByNotes.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_ROOT = osp.join(BASE_DIR, "data", "DATA_RAW", "root")


def load_token_embeddings(tokenizer: str = 'gatortron'):
    """
    사전 계산된 토큰 임베딩 로딩 함수.

    tokenizer 옵션:
      - 'clinicalbert' → clinicalbert_*.npy (예: clinicalbert_768.npy)
      - 'gatortron'    → gatortron_*.npy    (예: gatortron_1024.npy)
      - 'word2vec'     → word2vec_100 (gensim Word2Vec 포맷)

    clinicalbert/gatortron용 .npy 파일 형식:
      { token(str): np.ndarray[emb_dim] } 형태의 dict
    """
    emb_root = DATA_RAW_ROOT

    # clinicalbert / gatortron: npy dict 로딩
    if tokenizer in ['clinicalbert', 'gatortron']:
        if tokenizer == 'clinicalbert':
            prefix = 'clinicalbert_'
        else:  # gatortron
            prefix = 'gatortron_'

        pattern = osp.join(emb_root, f"{prefix}*.npy")
        candidates = glob.glob(pattern)
        if len(candidates) == 0:
            raise FileNotFoundError(f"[ERROR] No embedding file found matching: {pattern}")
        if len(candidates) > 1:
            print(f"[WARN] Multiple embedding files found for prefix '{prefix}'. "
                  f"Using the first one: {candidates[0]}")

        emb_path = candidates[0]
        print(f"[INFO] Loading pretrained {tokenizer} embeddings from {emb_path} ...")

        emb_dict = np.load(emb_path, allow_pickle=True).item()

        # 임베딩 차원 자동 추출
        example_vec = next(iter(emb_dict.values()))
        emb_dim = int(example_vec.shape[-1])

        def _get_vec(w: str):
            v = emb_dict.get(w)
            if v is None:
                return np.zeros(emb_dim, dtype=np.float32)
            return v.astype(np.float32)

        return _get_vec, emb_dim

    # word2vec (백워드 호환)
    elif tokenizer == 'word2vec':
        w2v_path = osp.join(emb_root, 'word2vec_100')
        print(f"[INFO] Loading pretrained word2vec embeddings from {w2v_path} ...")
        w2v = Word2Vec.load(w2v_path)
        emb_dim = w2v.vector_size

        def _get_vec(w: str):
            if w in w2v.wv.key_to_index:
                return w2v.wv[w].astype(np.float32)
            return np.zeros(emb_dim, dtype=np.float32)

        return _get_vec, emb_dim

    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")


class ConstructDatasetByNotes():
    """
    note-level 하이퍼그래프를 구성하고,
    각 노드에 (token-level) 임베딩을 붙여서 PyG Data 리스트를 생성하는 클래스.
    """
    def __init__(self, pre_path, split, dictionary, task, tokenizer='gatortron'):
        """
        Args:
            pre_path : data/DATA_PRE
            split    : 'train' or 'test'
            dictionary : vocab.txt에서 읽은 토큰 리스트
            task     : 'in-hospital-mortality'
            tokenizer : 'word2vec' or 'clinicalbert' or 'gatortron'
        """
        self.pre_path = pre_path
        self.split = split
        self.dictionary = dictionary
        self.task = task
        self.tokenizer = tokenizer
        super(ConstructDatasetByNotes, self).__init__()
        self.labels = self.get_labels(split)
        self.cat_path = osp.join(self.pre_path, 'categories.txt')

    def get_labels(self, split):
        """
        listfile.csv에서 patient_episode별 y_true (레이블) 로딩.
        """
        label_patients = pd.read_csv(
            osp.join(self.pre_path, self.task, split + '_hyper', 'listfile.csv'),
            sep=',',
            header=0
        )
        label_patients['name'] = label_patients.apply(
            lambda x: str(x['patient']) + '_' + x['episode'],
            axis=1
        )
        label_patients = label_patients.loc[:, ['name', 'y_true']]
        return label_patients

    def make_embedding(self, G, node, node_type, emb_dim):
        """
        node_emb vector 템플릿 생성.

        0: node_type -> {0:word, 1:note, 2:taxonomy}
        1: word_id   (vocab index) - word 노드일 때만 의미 있음
        2: note_id
        3: taxonomy_id (cat_id)
        4:~: token-level 임베딩 (word2vec / ClinicalBERT / GatorTronS 등)
        """
        emb = np.zeros(4 + emb_dim, dtype=np.float32)

        if node_type == 'word':
            emb[0] = 0
            emb[1] = -1
            emb[2] = G.nodes[node]['note_id']
            emb[3] = G.nodes[node]['cat_id']
        elif node_type == 'note':
            emb[0] = 1
            emb[1] = -1
            emb[2] = G.nodes[node]['note_id']
            emb[3] = G.nodes[node]['cat_id']
        elif node_type == 'tax':
            emb[0] = 2
            emb[1] = -1
            emb[2] = -1
            emb[3] = G.nodes[node]['cat_id']
        return emb

    def create_all_cats(self, path):
        """
        모든 train/test hyper 파일에서 CATEGORY 모아서 categories.txt 만든는 유틸 (원 코드 유지용).
        """
        all_cats = []
        for split in ['train', 'test']:
            hyper_path = osp.join(self.pre_path, self.task, split + '_hyper')
            patients = list(
                filter(lambda x: x in os.listdir(hyper_path),
                       list(self.labels['name']))
            )
            for patient in tqdm(patients[:], desc='Iterating over patients in {}_hyper'.format(split)):
                p_df = pd.read_csv(osp.join(hyper_path, patient), sep='\t', header=0)
                all_cats += p_df['CATEGORY'].tolist()
        all_cats = list(set(all_cats))
        f = open(f'{path}/categories.txt', 'w')
        f.write('\n'.join(all_cats))
        f.close()

    def set_node_embedding(self, G, node_attr='node_emb', get_vec=None, emb_dim=768):
        """
        노드에 node_emb 속성 채우기 (clinicalBERT/gatortron/word2vec 공용).
        """
        for node in G:
            node_name = str(node)

            if node_attr == 'node_emb':
                # 노트 노드 (n_으로 시작)
                if 'n_' in node_name:
                    emb = self.make_embedding(G, node_name, node_type='note', emb_dim=emb_dim)

                # 단어 노드 (taxonomy prefix 't_' 가 아닌 경우)
                elif not node_name.startswith('t_'):
                    emb = self.make_embedding(G, node_name, node_type='word', emb_dim=emb_dim)

                    # dictionary에서 단어 인덱스 찾기 (없으면 -1)
                    try:
                        emb[1] = self.dictionary.index(node_name)
                    except ValueError:
                        emb[1] = -1

                    # 단어 벡터 설정 (GatorTronS / ClinicalBERT / word2vec)
                    if get_vec is not None:
                        emb[4:] = get_vec(node_name)
                    else:
                        emb[4:] = np.zeros(emb_dim, dtype=np.float32)

                # taxonomy 노드
                else:
                    emb = self.make_embedding(G, node_name, node_type='tax', emb_dim=emb_dim)

            elif node_attr == 'pe':
                # positional encoding 등 (원 코드 호환용, 실제로는 사용 X)
                emb = node_attr[node_attr[:, 0] == node_name, 1:][0]
                assert (emb.astype(np.float32) == 1).sum() > 0

            else:
                raise ValueError('unknown node attribute')

            G.nodes[node_name][node_attr] = emb

        return G

    ### HyperGraph ###
    def construct_hypergraph_datalist(self):
        """
        train_hyper / test_hyper 아래 episode별 hyper 파일을 읽어서
        patient-level 하이퍼그래프 목록을 생성.
        """
        print()
        print("<<Start Construct Hypergraph Datalist>>")

        # 토큰 임베딩 로딩 (GatorTronS / ClinicalBERT / word2vec)
        get_vec, emb_dim = load_token_embeddings(self.tokenizer)

        # hypergraph source dir (episode별 .csv)
        hyper_path = osp.join(self.pre_path, self.task, self.split + '_hyper')

        # 1) listfile.csv 기준 episode 이름들 (확장자 없음)
        names_from_label = list(self.labels['name'])

        # 2) 실제 디스크에 있는 파일들 (확장자 포함)
        files_on_disk = os.listdir(hyper_path)

        # 3) 실제 파일이 존재하는 episode만 유지
        patients = []
        for name in names_from_label:
            if f"{name}.csv" in files_on_disk:
                patients.append(name)

        # 4) 사용할 주요 카테고리 6개만 정의
        CATEGORY_LIST = ['Radiology', 'Nursing', 'Nursing/other', 'ECG', 'Echo', 'Physician']
        cat_to_id = {cat: i for i, cat in enumerate(CATEGORY_LIST)}

        print('<Patient list generation done>')
        Data_list = []

        for patient in tqdm(patients[:], desc='Iterating over patients in {}_hyper'.format(self.split)):
            # ex) /.../train_hyper/3_episode145834.csv
            p_df = pd.read_csv(
                osp.join(hyper_path, f"{patient}.csv"),
                sep='\t',
                header=0
            )

            # 기본 클리닝
            p_df = p_df.dropna(axis=0)

            # CATEGORY 문자열 정리
            p_df['CATEGORY'] = p_df['CATEGORY'].astype(str).str.strip()

            # 이 6개 카테고리만 사용
            if len(p_df[p_df['CATEGORY'].isin(CATEGORY_LIST)]) == 0:
                continue
            p_df = p_df[p_df['CATEGORY'].isin(CATEGORY_LIST)]

            # note 30개 초과면 컷
            if p_df['note_id'].nunique() > 30:
                max_nid = p_df['note_id'].unique()[30]
                p_df = p_df[p_df['note_id'] < max_nid]

            # 타입 통일
            p_df['WORD'] = p_df['WORD'].astype(str)
            p_df['SENT'] = p_df['SENT'].astype(str)
            p_df['note_id'] = p_df['note_id'].astype(str)
            p_df['note_NM'] = "n_" + p_df['note_id']

            # patient-level label (hospital mortality)
            y_p = self.labels[self.labels['name'] == patient]['y_true'].values[0]
            y_p = torch.from_numpy(np.array([y_p])).to(torch.long)

            G_n_list = []
            y_n_list = []
            hour_n_list = []

            # 노트 단위로 그래프 구성
            for n_id, n_df in p_df.groupby(by='note_NM'):
                n_df = n_df.dropna(axis=0)
                n_df = n_df[n_df['WORD'] != 'nan']

                if len(n_df) == 0:
                    continue

                # word-note edge = 1
                n_df['edge_type'] = 1

                # note마다 bipartite graph
                G_n = nx.from_pandas_edgelist(n_df, 'WORD', 'note_NM', 'edge_type')

                # 단어 300개 이상이면 잘라냄
                cut_300 = list(G_n.nodes)[300:]
                G_n.remove_nodes_from(cut_300)

                # 카테고리 -> cat_id 로 매핑
                cat_str = n_df['CATEGORY'].values[0].strip()
                if cat_str not in cat_to_id:
                    continue
                cat_id = cat_to_id[cat_str]

                # note id
                note_id = int(n_df['note_id'].values[0])

                # 노드 속성 부여(note_id, cat_id)
                attrs = {}
                for node in G_n:
                    attrs[node] = {
                        'note_id': note_id,
                        'cat_id': cat_id
                    }
                nx.set_node_attributes(G_n, attrs)

                # 임베딩 붙이기 (word2vec / ClinicalBERT / GatorTronS 공용)
                G_n = self.set_node_embedding(
                    G_n,
                    node_attr='node_emb',
                    get_vec=get_vec,
                    emb_dim=emb_dim
                )
                G_n_list.append(G_n)

                y_n_list.append([cat_id])
                hour_n_list.append([n_df['Hours'].values[0]])

            # 노트가 하나도 안 남으면 이 patient 스킵
            if len(G_n_list) == 0:
                continue

            # 여러 note graph들을 하나의 그래프로 합침
            G_n = nx.disjoint_union_all(G_n_list)

            # taxonomy 노드 추가 & edge_type=2 부여
            for node in range(len(G_n)):
                # node_emb[0] == 0  → 'word' 노드
                if G_n.nodes[node]['node_emb'][0] == 0:
                    tax_node = 't_' + str(G_n.nodes[node]['cat_id'])

                    # word-taxonomy edge_type=2
                    G_n.add_edge(node, tax_node, edge_type=2)

                    # taxonomy 노드 초기화
                    if 'node_emb' not in G_n.nodes[tax_node]:
                        emb = self.make_embedding(G_n, node, node_type='tax', emb_dim=emb_dim)
                        G_n.nodes[tax_node]['node_emb'] = emb
                    if 'note_id' not in G_n.nodes[tax_node]:
                        G_n.nodes[tax_node]['note_id'] = G_n.nodes[node]['note_id']
                        G_n.nodes[tax_node]['cat_id'] = G_n.nodes[node]['cat_id']

            # graph → torch geometric 텐서들
            edge_index_n, edge_index_mask, x_n, batch_t, batch_n = graph_to_torch_sparse_tensor(
                G_n,
                node_attr=['note_id', 'cat_id']
            )

            y_n = torch.from_numpy(np.array(y_n_list)).to(torch.long)
            hour_n = torch.from_numpy(np.array(hour_n_list)).to(torch.float32)

            data = Data(
                x_n=x_n,
                edge_index_n=edge_index_n,
                edge_index_mask=edge_index_mask,
                hour_n=hour_n,
                y_n=y_n,
                y_p=y_p,
                batch_n=batch_n,
                batch_t=batch_t
            )
            Data_list.append(data)

        print('<Hypergraph Data list generation done>')
        print('<<End Construct Hypergraph Datalist>>')
        print()
        return Data_list


if __name__ == '__main__':
    # 로컬 테스트용 (서버에서 바로 실행해도 됨)
    task = 'in-hospital-mortality'

    BASE = BASE_DIR
    RAW_PATH = osp.join(BASE, 'data', 'DATA_RAW')
    PRE_PATH = osp.join(BASE, 'data', 'DATA_PRE')

    dictionary_path = osp.join(RAW_PATH, 'root', 'vocab.txt')
    dictionary = open(dictionary_path).read().split()

    cdbn = ConstructDatasetByNotes(
        pre_path=PRE_PATH,
        split='test',              # 또는 'train'
        dictionary=dictionary,
        task=task,
        tokenizer='gatortron'      # 🔴 GatorTronS 토큰 임베딩 사용
    )
    data_list = cdbn.construct_hypergraph_datalist()
    print(f"[INFO] Constructed {len(data_list)} graphs.")
