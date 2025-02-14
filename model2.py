import torch
import torch.nn as nn
import torch.nn.functional as F
from components.group import GroupOperation
from components.gcn import GCNBlock
from components.lstm import PointLSTM
from components.mlp import MLPBlock

class CompleteEnzymeModel(nn.Module):
    def __init__(self, task, use_gcn=True, use_lstm=True, use_quat=True):
        super(CompleteEnzymeModel, self).__init__()
        # [Previous init code remains the same...]
        self.task = task
        self.num_classes = task.num_classes
        self.use_gcn = use_gcn
        self.use_lstm = use_lstm
        self.use_quat = use_quat
        
        # Feature dimensions
        self.hidden_dim = 128
        self.embedding_dim = 128
        self.lstm_dim = 256
        
        # Node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(1, self.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.2)
        )
        
        # GCN layers
        if use_gcn:
            self.gcn_layers = nn.ModuleList([
                GCNBlock(d_model=self.embedding_dim, hidden_features=self.hidden_dim, k=2),
                GCNBlock(d_model=self.hidden_dim, hidden_features=self.hidden_dim, k=2)
            ])
        
        # LSTM module
        if use_lstm:
            self.feature_reduction = nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.LayerNorm(64),
                nn.Linear(64, 4),
                nn.ReLU()
            )
            
            self.lstm_proj = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.LayerNorm(64),
                nn.Linear(64, self.lstm_dim),
                nn.ReLU()
            )
            
            self.lstm = PointLSTM(
                pts_num=4,
                in_channels=self.lstm_dim,
                hidden_dim=self.hidden_dim,
                offset_dim=4,
                num_layers=1,
                topk=4,
                use_quat=use_quat
            )
        
        # Feature transformation blocks
        self.feature_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim * 2, 1),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Conv1d(self.hidden_dim * 2, self.hidden_dim * 4, 1),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        ])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
        
        # EC number mapping
        self.ec_map = {str(i): i-1 for i in range(1, 8)}
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)

    def forward(self, data):
        # [Forward method remains the same...]
        device = next(self.parameters()).device
        
        # Node feature processing
        x = data.x.float().unsqueeze(-1)
        x = self.node_embedding(x)
        
        batch_size = int(data.batch.max()) + 1
        outputs = []
        
        for i in range(batch_size):
            # Get nodes for current graph
            mask = (data.batch == i)
            graph_nodes = x[mask]
            
            # Process features
            current_features = graph_nodes.t()
            
            # GCN processing
            if self.use_gcn:
                current_features = current_features.t()
                mask_idx = mask.nonzero().squeeze()
                start_idx = mask_idx[0].item()
                end_idx = mask_idx[-1].item() + 1
                edges_mask = (data.edge_index[0] >= start_idx) & (data.edge_index[0] < end_idx)
                graph_edges = data.edge_index[:, edges_mask]
                graph_edges = graph_edges - start_idx
                
                for gcn in self.gcn_layers:
                    current_features = gcn(current_features, graph_edges)
            
            # LSTM processing
            if self.use_lstm:
                features = current_features
                features = self.feature_reduction(features)
                
                features = features.t().unsqueeze(0)
                features = F.adaptive_max_pool1d(features, 4)
                
                lstm_input = self.lstm_proj(features.permute(0, 2, 1))
                lstm_input = lstm_input.permute(0, 2, 1).unsqueeze(-1)
                
                lstm_outputs, _, _ = self.lstm(lstm_input)
                current_features = lstm_outputs[-1][0].squeeze(-1)
            
            # Feature transformation
            current_features = current_features.unsqueeze(0)
            
            for block in self.feature_blocks:
                current_features = block(current_features)
                current_features = F.instance_norm(current_features)
            
            # Global pooling
            graph_embedding = self.global_pool(current_features).squeeze(-1).squeeze(0)
            outputs.append(graph_embedding)
        
        # Final classification
        x = torch.stack(outputs)
        logits = self.classifier(x)
        
        return logits

    def train_step(self, batch):
        self.train()
        self.optimizer.zero_grad()
        device = next(self.parameters()).device
        
        try:
            # Process data and get labels
            if isinstance(batch, list):
                data = self.move_graph_to_device(batch[0], device)
                protein_data = batch[1]['protein']
                labels = torch.tensor([self._get_label_from_ec(ec) for ec in protein_data['EC']], 
                                    dtype=torch.long, device=device)
            else:
                data = self.move_graph_to_device(batch, device)
                labels = batch.y.to(device)
            
            # Forward pass and loss computation
            logits = self(data)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            return {"loss": loss.item()}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"loss": 0.0}

    def test_step(self, batch_data):
        self.eval()
        device = next(self.parameters()).device
        
        try:
            # Handle both list and single batch formats
            if isinstance(batch_data, list):
                data = self.move_graph_to_device(batch_data[0], device)
            else:
                data = self.move_graph_to_device(batch_data, device)
            
            with torch.no_grad():
                logits = self(data)
                probs = F.softmax(logits, dim=-1)
                # Move to CPU for numpy conversion
                return probs.cpu()
        except Exception as e:
            import traceback
            traceback.print_exc()
            if isinstance(batch_data, list):
                batch_size = int(batch_data[0].batch.max()) + 1
            else:
                batch_size = int(batch_data.batch.max()) + 1 if hasattr(batch_data, 'batch') else 1
            return torch.zeros(batch_size, self.num_classes, device='cpu')

    def _get_label_from_ec(self, ec_number: str) -> int:
        try:
            first_number = ec_number.split('.')[0]
            return self.ec_map.get(first_number, 0)
        except:
            return 0

    def move_graph_to_device(self, data, device):
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.to(device)
        if hasattr(data, 'batch'):
            data.batch = data.batch.to(device)
        return data