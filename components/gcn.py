# models/gcn.py
import torch
from torch import nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        x = torch.matmul(x, self.weight)
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col])
        return out + self.bias

class GCNBlock(nn.Module):
    def __init__(self, d_model, hidden_features, k=32):
        super(GCNBlock, self).__init__()
        self.k = k
        self.gc1 = GraphConvolution(d_model, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, d_model)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, edge_index=None):
        device = x.device
        
        # If we receive PyG format data
        if edge_index is not None:
            # Apply GCN layers
            residual = x
            x = self.gc1(x, edge_index)
            x = F.relu(self.norm1(x))
            x = self.gc2(x, edge_index)
            x = self.norm2(x)
            x = x + residual
            return x
            
        # If we receive the old format data
        else:
            batch, d_model, timestep, num_points = x.shape
            
            # Reshape x to (batch * timestep, num_points, d_model)
            x = x.permute(0, 2, 3, 1).contiguous().view(-1, num_points, d_model)
            
            # Get k-nearest neighbors
            edge_index = self.get_graph_feature_efficient(x)
            edge_index = edge_index.to(device)
            
            # Reshape x to (batch * timestep * num_points, d_model)
            x = x.view(-1, d_model)
            
            # Apply GCN layers
            residual = x
            x = self.gc1(x, edge_index)
            x = F.relu(self.norm1(x))
            x = self.gc2(x, edge_index)
            x = self.norm2(x)
            x = x + residual
            
            # Reshape back to (batch, d_model, timestep, num_points)
            x = x.view(batch, timestep, num_points, d_model).permute(0, 3, 1, 2)
            
            return x

    def get_graph_feature_efficient(self, x):
        """Memory-efficient k-NN graph construction"""
        batch_size, num_points, d_model = x.shape
        device = x.device
        
        chunk_size = 128
        edge_indices = []
        
        for i in range(0, batch_size * num_points, chunk_size):
            end_idx = min(i + chunk_size, batch_size * num_points)
            x_chunk = x.view(-1, d_model)[i:end_idx]
            dist_chunk = torch.cdist(x_chunk, x.view(-1, d_model), p=2)
            _, nn_idx = torch.topk(-dist_chunk, k=min(self.k, dist_chunk.size(1)), dim=1)
            
            rows = torch.arange(i, end_idx, device=device).view(-1, 1).repeat(1, min(self.k, dist_chunk.size(1))).view(-1)
            cols = nn_idx.view(-1)
            edge_indices.append(torch.stack([rows, cols], dim=0))
            
            del dist_chunk, nn_idx
            torch.cuda.empty_cache()
        
        edge_index = torch.cat(edge_indices, dim=1)
        return edge_index.to(device)