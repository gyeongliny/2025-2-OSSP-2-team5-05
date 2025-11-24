import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


##################################
########### TM-HGNN #############
##################################

class TM_HGNN(torch.nn.Module):
    """
    TM-HGNN with projection:
    - 입력 특징 = [meta(4) + embedding_dim]
    - emb_dim → proj_dim (기본 256)
    - 이후 GCNConv 3층, edge_mask로 다른 hyperedge 별로 propagation
    - Global pooling 후 Binary classification
    """

    def __init__(self, num_features, hidden_channels,
                 proj_dim=256, dropout=0.3):
        """
        Args:
            num_features : 4 + emb_dim  (clinicalBERT: 772, GatorTron: 1028 등)
            hidden_channels : GCNConv hidden size
            proj_dim : embedding projection dimension (default: 256)
            dropout : dropout ratio
        """
        super(TM_HGNN, self).__init__()

        # meta + emb 분리
        self.meta_dim = 4
        self.emb_dim  = num_features - self.meta_dim
        self.proj_dim = proj_dim
        self.dropout  = dropout

        # 1) 임베딩 projection layer
        self.proj = Linear(self.emb_dim, self.proj_dim)      # emb_dim → 256

        # 2) 첫 번째 GCN 입력 = meta(4) + proj_dim(256)
        conv_in_dim = self.meta_dim + self.proj_dim

        # 3) Multi-hyperedge GCN layers
        self.conv1 = GCNConv(conv_in_dim, hidden_channels)
        self.bn1   = BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2   = BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3   = BatchNorm1d(hidden_channels)

        # 4) Final layer
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_mask, batch):
        """
        x : [N, num_features = 4 + emb_dim]
        edge_index : [2, num_edges]
        edge_mask : 각 edge의 타입  (1=Note, 2=Taxonomy)
        batch : [N]  → 그래프별 pooling용 batch index
        """

        # 1) meta + embedding 분리
        x_meta = x[:, :self.meta_dim]             # [N, 4]
        x_emb  = x[:, self.meta_dim:]             # [N, emb_dim]

        # 2) 임베딩 projection
        x_emb_proj = self.proj(x_emb)             # [N, proj_dim]

        # 3) 다시 concat → GCN 입력
        x = torch.cat([x_meta, x_emb_proj], dim=-1)

        # --------------------------
        # 1) All hyperedges propagation
        # --------------------------
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # --------------------------
        # 2) Note hyperedges only
        # --------------------------
        idx_1 = torch.where(edge_mask == 1)[0]  # Note edges
        if idx_1.numel() > 0:
            x = self.conv2(x, edge_index[:, idx_1])
            x = self.bn2(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # --------------------------
        # 3) Taxonomy hyperedges only
        # --------------------------
        idx_2 = torch.where(edge_mask == 2)[0]  # Taxonomy edges
        if idx_2.numel() > 0:
            x = self.conv3(x, edge_index[:, idx_2])
            x = self.bn3(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # --------------------------
        # Graph-level pooling
        # --------------------------
        x = global_mean_pool(x, batch)           # [batch_size, hidden_channels]

        # --------------------------
        # Final prediction layer
        # --------------------------
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x


if __name__ == '__main__':
    print("TM-HGNN (GPT-ready version)")
