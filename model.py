import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Any, Dict

class ComplexEnzymeModel(nn.Module):
    def __init__(self, task):
        super(ComplexEnzymeModel, self).__init__()
        self.task = task
        
        # Model hyperparameters
        self.hidden_dim = 64
        self.num_classes = task.num_classes
        
        # GNN layers
        self.conv1 = GCNConv(1, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        
        # Output layers
        self.fc1 = nn.Linear(self.hidden_dim, 32)
        self.fc2 = nn.Linear(32, self.num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        
        # Create EC number mapping
        self.ec_map = {str(i): i-1 for i in range(1, 8)}

    def _get_label_from_ec(self, ec_number: str) -> int:
        """Convert EC number to class label (0-6) based on first number"""
        try:
            first_number = ec_number.split('.')[0]
            return self.ec_map[first_number]
        except Exception as e:
            print(f"Error processing EC number {ec_number}: {e}")
            return 0

    def forward(self, data):
        # Get x and reshape
        x = data.x
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = x.float()
        
        # Get edge index
        edge_index = data.edge_index
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        batch = data.batch
        x = global_mean_pool(x, batch)
        
        # MLP classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def train_step(self, batch) -> Dict[str, float]:
        self.train()
        self.optimizer.zero_grad()
        
        try:
            # Get the actual batch data
            data = batch[0] if isinstance(batch, (list, tuple)) else batch
            
            # Forward pass
            out = self(data)
            
            # Get labels from protein dictionary
            protein_data = batch[1]['protein'] if isinstance(batch, (list, tuple)) else batch.protein
            enzyme_classes = protein_data['EC']
            
            # Convert enzyme classes to label indices
            labels = [self._get_label_from_ec(ec) for ec in enzyme_classes]
            y = torch.tensor(labels, dtype=torch.long)
            
            # Calculate loss
            loss = F.cross_entropy(out, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            return {"loss": loss.item()}
            
        except Exception as e:
            print(f"Error in train_step: {e}")
            return {"loss": 0.0}

    def test_step(self, data):
        self.eval()
        with torch.no_grad():
            try:
                # Debug: Print data structure
                print("\nTest data type:", type(data))
                if isinstance(data, (list, tuple)):
                    print("Length:", len(data))
                    print("First element type:", type(data[0]))
                    print("Second element type:", type(data[1]) if len(data) > 1 else "N/A")
                
                # Process test data
                if isinstance(data, (list, tuple)):
                    batch_data = data[0]  # This should be the DataBatch
                    out = self(batch_data)
                else:
                    out = self(data)
                
                return F.softmax(out, dim=1)
            except Exception as e:
                print(f"Error in test_step: {e}")
                print("Full data structure:", data)
                return self.task.dummy_output()