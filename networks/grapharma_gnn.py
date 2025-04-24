# networks/grapharma_gnn.py
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, gnn_type='GCN'):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.convs = nn.ModuleList()
        self.convs.append(self._create_gnn_layer(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(self._create_gnn_layer(hidden_channels, hidden_channels))
        self.out = nn.Linear(hidden_channels, 1)

    def _create_gnn_layer(self, in_channels, out_channels):
        if self.gnn_type == 'GCN':
            return GCNConv(in_channels, out_channels)
        elif self.gnn_type == 'SAGE':
            return SAGEConv(in_channels, out_channels)
        elif self.gnn_type == 'GAT':
            return GATConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.out(x).view(-1)
