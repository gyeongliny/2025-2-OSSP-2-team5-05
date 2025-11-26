import torch
from torch import nn
import numpy as np

torch.manual_seed(42)


class Handle_data(object):
    """
    TM-HGNN 입력용 전처리 클래스.
    - data.x_n: [N, 4 + emb_dim]
        * 0: node type (예: 0=word, 1=note hyperedge, 2=taxonomy hyperedge ...)
        * 1: index (노트 index, taxonomy index 등)
        * 2,3: 기타 메타 정보 (예: 시간 등)
        * 4~: 사전 계산된 임베딩 (word2vec / ClinicalBERT / GatorTronS 등)
    - 이 클래스는:
        * x_type, x_idx 설정
        * 마지막 emb_dim 부분을 data.x로 사용
        * taxonomy / note / hour embedding을 같은 차원으로 생성 후 더해줌

    👉 emb_dim 을 data.x_n.size(1) - 4 로 자동 계산하기 때문에
       word2vec(100), ClinicalBERT(768), GatorTronS(1024) 전부 호환됨.
    """

    def __init__(self, type, tokenizer,
                 num_taxonomy: int = 15,
                 num_notes: int = 100,
                 hour_bins=(0.0, 24.0, 48.0)):
        self.type = type
        self.tokenizer = tokenizer

        # taxonomy, note, hour embedding 크기 설정
        self.num_taxonomy = num_taxonomy
        self.num_notes = num_notes
        self.hour_bins = hour_bins  # (0.0, 24.0, 48.0)

        # Embedding 레이어는 emb_dim을 알아야 만들 수 있으므로
        # 첫 호출 때 lazy-init 한다.
        self._t_emb = None
        self._n_emb = None
        self._h_emb = None

    def _ensure_embeddings(self, emb_dim: int, device: torch.device):
        """
        emb_dim을 보고 taxonomy/note/hour embedding 레이어를 생성 (또는 재사용).
        GatorTronS로 emb_dim=1024이든, word2vec으로 100이든 자동으로 맞춰준다.
        """
        # taxonomy hyperedge embedding
        if (self._t_emb is None) or (self._t_emb.embedding_dim != emb_dim):
            self._t_emb = nn.Embedding(self.num_taxonomy, emb_dim)
            self._t_emb.weight.requires_grad = False

        # note hyperedge embedding
        if (self._n_emb is None) or (self._n_emb.embedding_dim != emb_dim):
            self._n_emb = nn.Embedding(self.num_notes, emb_dim)
            self._n_emb.weight.requires_grad = False

        # hour embedding (0, 24, 48 구간 → 3개 bin)
        if (self._h_emb is None) or (self._h_emb.embedding_dim != emb_dim):
            self._h_emb = nn.Embedding(3, emb_dim)
            self._h_emb.weight.requires_grad = False

        # 모두 device에 맞춰주기 (보통 CPU)
        self._t_emb.to(device)
        self._n_emb.to(device)
        self._h_emb.to(device)

    def tokenize(self, data):
        # edge mask 재이름 지정
        data.edge_mask = data.edge_index_mask
        del data.edge_index_mask

        # type / index
        # x_n: [N, 4 + emb_dim]
        data.x_type = data.x_n[:, 0].to(torch.long)  # node type
        data.x_idx = data.x_n[:, 1].to(torch.long)   # node index 등

        # ==== 임베딩 차원 자동 계산 ====
        # 앞 4개 컬럼은 meta, 뒤는 임베딩
        emb_dim = data.x_n.size(1) - 4
        device = data.x_n.device

        # 원래 임베딩 (word/GatorTronS 등)을 x에 복사
        data.x = data.x_n[:, 4:4 + emb_dim].to(torch.float32)

        # taxonomy / note / hour embedding 레이어 준비
        self._ensure_embeddings(emb_dim, device)

        # ============================
        # TAXONOMY Hyperedge embedding
        # ============================
        # x_type == 2 인 노드 위치
        t_idx = (data.x_type == 2).nonzero(as_tuple=False).view(-1)
        if t_idx.numel() > 0:
            # data.batch_t: taxonomy index (각 노드마다 0~num_taxonomy-1)
            # t_idx 위치에 해당하는 batch_t 인덱스만 뽑아서 embedding
            batch_t_for_nodes = torch.index_select(data.batch_t, 0, t_idx.to(data.batch_t.device))
            tax_emb = self._t_emb(batch_t_for_nodes.to(device)).to(torch.float32)
            data.x[t_idx.long(), :] = tax_emb

        # ============================
        # NOTE Hyperedge + hour embedding
        # ============================
        # x_type == 1 인 노드 위치
        n_idx = (data.x_type == 1).nonzero(as_tuple=False).view(-1)
        if n_idx.numel() > 0:
            # NOTE hyperedge embedding
            batch_n_for_nodes = torch.index_select(data.batch_n, 0, n_idx.to(data.batch_n.device))
            note_emb = self._n_emb(batch_n_for_nodes.to(device)).to(torch.float32)

            # hour embedding (0, 24, 48 구간)
            # data.hour_n: [N, 1] or [N]
            dic = {0.0: 0, 24.0: 1, 48.0: 2}

            # hour_n 이 GPU에 있어도 안전하게 CPU로 가져와서 numpy 처리
            hours = data.hour_n.squeeze(-1).detach().cpu().numpy()
            binned_keys = []
            for h in hours:
                if h < self.hour_bins[1]:     # < 24.0
                    key = self.hour_bins[0]   # 0.0
                elif h < self.hour_bins[2]:   # < 48.0
                    key = self.hour_bins[1]   # 24.0
                else:
                    key = self.hour_bins[2]   # 48.0
                binned_keys.append(key)

            indices = [dic[k] for k in binned_keys]
            indices = torch.tensor(indices, dtype=torch.long, device=device)
            hour_emb_full = self._h_emb(indices)  # [N, emb_dim]

            # note 노드 위치에 해당하는 hour embedding만 추출
            hour_emb = torch.index_select(hour_emb_full, 0, n_idx.to(device))

            # NOTE hyperedge embedding + hour embedding 더해서 적용
            data.x[n_idx.long(), :] = note_emb + hour_emb

        # 라벨
        data.y = data.y_p.to(torch.float32)
        return data

    def __call__(self, data):
        return self.tokenize(data)
