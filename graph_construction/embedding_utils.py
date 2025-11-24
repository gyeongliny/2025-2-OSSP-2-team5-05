import torch
import torch.nn
from torch.nn import Embedding
import torch.nn.functional as F
torch.manual_seed(42)

import numpy as np


class Handle_data(object):
    def __init__(self, type, tokenizer):
        self.type = type
        self.tokenizer = tokenizer

    def tokenize(self, data):
        data.edge_mask = data.edge_index_mask
        del data.edge_index_mask            

        # type / index
        data.x_type = data.x_n[:, 0].to(torch.long)
        data.x_idx  = data.x_n[:, 1].to(torch.long)

        # ==== 임베딩 차원 자동 계산 ====
        # x_n: [N, 4 + emb_dim]  → 앞 4개는 meta, 뒤가 임베딩
        emb_dim = data.x_n.size(1) - 4
        data.x  = data.x_n[:, 4:4+emb_dim].to(torch.float32)

        # ==== hyperedge embedding 도 emb_dim에 맞추기 ====
        # TAXONOMY Hyperedge
        t_idx = (data.x_type == 2).nonzero().squeeze()
        t_emb = Embedding(15, emb_dim)
        t_emb.weight.requires_grad = False
        tax_emb = t_emb(torch.index_select(data.batch_t, 0, t_idx)).to(torch.float32)
        data.x[t_idx.long(), :] = tax_emb

        # NOTE Hyperedge
        n_idx = (data.x_type == 1).nonzero().squeeze()
        n_emb = Embedding(100, emb_dim)
        n_emb.weight.requires_grad = False
        note_emb = n_emb(torch.index_select(data.batch_n, 0, n_idx)).to(torch.float32)

        # hour embedding (0, 24, 48 구간)
        dic = {0.0: 0, 24.0: 1, 48.0: 2}
        h_emb = Embedding(3, emb_dim)
        h_emb.weight.requires_grad = False

        hours = data.hour_n.squeeze(-1).numpy()
        binned_keys = []
        for h in hours:
            if h < 24.0:
                key = 0.0
            elif h < 48.0:
                key = 24.0
            else:
                key = 48.0
            binned_keys.append(key)

        indices = [dic[k] for k in binned_keys]
        indices = torch.tensor(indices, dtype=torch.long)
        hour_emb = h_emb(indices)

        data.x[n_idx.long(), :] = note_emb + hour_emb

        data.y = data.y_p.to(torch.float32)
        return data
    
    def __call__(self, data):
        return self.tokenize(data)
