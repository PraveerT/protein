import torch
import torchvision.models.video as models
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
import os
NUM_PROTEINS = 20

def custom_collate(batch):
    inputs = torch.stack([item[0] for item in batch])
    metadata = [item[1] for item in batch]
    return inputs, metadata

class VoxelEnzymeModel(nn.Module):
    def __init__(self, task):
        super(VoxelEnzymeModel, self).__init__()
        self.task = task
        self.swin_encoder = models.swin3d_b(weights="KINETICS400_V1")
        self.swin_encoder.patch_embed.proj = nn.Conv3d(NUM_PROTEINS, 128, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.classifier = nn.Linear(self.swin_encoder.head.out_features, task.num_classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.ec_map = {str(i): i-1 for i in range(1, 8)}
        
    def _get_label_from_ec(self, ec_number):
        try:
            first_number = ec_number.split('.')[0]
            return self.ec_map[first_number]
        except:
            return 0
        
    def forward(self, data):
        x = self.swin_encoder(data)
        x = self.classifier(x)
        return x
    
    def train_step(self, train_dataset, test_dataset=None, batch_size=8, lr=1e-4, num_epochs=10, optimizer="Adam", save_dir="models"):
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.to(self.device)
        optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=lr)
        

        train_labels = self.task.train_targets
        class_counts = Counter(train_labels)
        total_samples = sum(class_counts.values())

        weights = {cls: total_samples / count for cls, count in class_counts.items()}
        weight_tensor = torch.tensor(
            [weights.get(i, 1) for i in range(self.task.num_classes)], 
            dtype=torch.float
        ).to(self.device)

        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate, shuffle=True)

        best_loss = float("inf")

        for epoch in range(num_epochs):
            self.train()
            epoch_losses = []

            for batch in train_loader:
                inputs, metadata = batch
                optimizer.zero_grad()
                inputs = inputs.permute(0, 4, 1, 2, 3).to(self.device)

                labels = torch.tensor(
                    [self._get_label_from_ec(meta['protein']['EC']) for meta in metadata],
                    dtype=torch.long
                ).to(self.device)
                
                output = self(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.state_dict(), os.path.join(save_dir, "best_model.pth"))
                print(f"Best model saved with loss {best_loss:.4f}")

            torch.save(self.state_dict(), os.path.join(save_dir, "last_model.pth"))

            if test_dataset:
                metrics, _ = self.eval_step(test_dataset)
                print(f"Evaluation at Epoch {epoch+1}: {metrics}")

    def eval_step(self, test_dataset, batch_size = 8):
        self.to(self.device)
        self.eval()
        result = []

        with torch.no_grad():
            for batch in DataLoader(test_dataset, batch_size=batch_size,  collate_fn=custom_collate):
                inputs, _ = batch
                inputs = inputs.permute(0, 4, 1, 2, 3).to(self.device)

                output = self(inputs)
                predicted = torch.argmax(output, dim=1).cpu().numpy()
                result.append(predicted)

        metrics = self.task.evaluate(self.task.test_targets, np.concatenate(result))
        return metrics, np.concatenate(result)
