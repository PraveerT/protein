from proteinshake.tasks import EnzymeClassTask
from torch_geometric.loader import DataLoader
from model import ComplexEnzymeModel
import numpy as np
import torch

# Use proteins with Enzyme Class annotations
task = EnzymeClassTask().to_graph(eps=8).pyg()

# Create model
model = ComplexEnzymeModel(task)

# Training using native data loaders
for epoch in range(20):
    # Training
    model.train()
    epoch_losses = []
    
    for batch in DataLoader(task.train, batch_size=32, shuffle=True):
        loss_info = model.train_step(batch)
        epoch_losses.append(loss_info["loss"])
    
    # Calculate average loss for epoch
    avg_loss = np.mean(epoch_losses)
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

# Final evaluation with the provided metrics
prediction = model.test_step(task.test)
metrics = task.evaluate(task.test_targets, prediction)

print("\nFinal Test Metrics:", metrics)