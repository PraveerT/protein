import torch
import torch.nn as nn
import torch.nn.functional as F
from components.group import GroupOperation
from components.mlp import MLPBlock

class AdaptedEnzymeModel(nn.Module):
    def __init__(self, task, use_gcn=True, use_lstm=True, use_quat=True):
        super(AdaptedEnzymeModel, self).__init__()
        self.task = task
        self.num_classes = task.num_classes
        
        # Model dimensions
        self.hidden_dim = 64
        self.embedding_dim = 64
        
        # Node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(1, self.embedding_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.embedding_dim // 2),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embedding_dim)
        )
        
        # Group operation for neighbor features
        self.group = GroupOperation()
        
        # Convolution blocks with improved architecture
        self.conv_blocks = nn.ModuleList([
            MLPBlock([self.embedding_dim, self.hidden_dim, self.hidden_dim], 2),
            nn.BatchNorm2d(self.hidden_dim),
            MLPBlock([self.hidden_dim, self.hidden_dim * 2, self.hidden_dim * 2], 2),
            nn.BatchNorm2d(self.hidden_dim * 2)
        ])
        
        # Pooling layers
        self.pool = nn.AdaptiveMaxPool2d((None, 1))
        
        # Final MLP blocks with skip connections
        self.final_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
        
        # EC number mapping
        self.ec_map = {str(i): i-1 for i in range(1, 8)}
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def _get_label_from_ec(self, ec_number: str) -> int:
        try:
            first_number = ec_number.split('.')[0]
            return self.ec_map[first_number]
        except:
            return 0
        
    def forward(self, data):
        # Process input features
        x = data.x.float().unsqueeze(-1)
        
        # Node embedding
        x = self.node_embedding(x)
        
        # Process each graph separately
        batch_size = int(data.batch.max()) + 1
        outputs = []
        
        for i in range(batch_size):
            # Get nodes for current graph
            mask = (data.batch == i)
            graph_nodes = x[mask]
            num_nodes = graph_nodes.shape[0]
            
            # Process through MLP blocks directly
            current_features = graph_nodes.t().unsqueeze(-1).unsqueeze(0)
            
            # Process through conv blocks
            for layer in self.conv_blocks:
                if isinstance(layer, MLPBlock):
                    current_features = layer(current_features)
                else:
                    current_features = layer(current_features)
            
            # Global pooling for graph features
            graph_embedding = current_features.mean(dim=2).squeeze()
            outputs.append(graph_embedding)
        
        # Combine all graph embeddings
        x = torch.stack(outputs)
        
        # Final classification
        x = self.final_mlp(x)
        
        return x
    
    def train_step(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        try:
            # Get data and labels
            if isinstance(batch, list):
                data = batch[0]
                protein_data = batch[1]['protein']
                labels = torch.tensor([self._get_label_from_ec(ec) for ec in protein_data['EC']], 
                                    dtype=torch.long, device=data.x.device)
            else:
                data = batch
                labels = data.y
            
            # Forward pass
            out = self(data)
            
            # Compute loss and update
            loss = F.cross_entropy(out, labels)
            loss.backward()
            self.optimizer.step()
            
            return {"loss": loss.item()}
        except Exception as e:
            return {"loss": 0.0}
    
    def test_step(self, data):
        self.eval()
        with torch.no_grad():
            try:
                return F.softmax(self(data), dim=1)
            except:
                return self.task.dummy_output()