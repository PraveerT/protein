# main.py
import torch
from proteinshake.tasks import EnzymeClassTask
from torch_geometric.loader import DataLoader
from model2 import CompleteEnzymeModel

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create task with protein data
    task = EnzymeClassTask().to_graph(eps=8).pyg()
    
    print(f"Dataset sizes: Train {len(task.train)}, Test {len(task.test)}")

    # Create model
    model = CompleteEnzymeModel(
        task=task,
        use_gcn=True,
        use_lstm=True,
        use_quat=True
    ).to(device)

    # Training loop
    print("Starting training...")
    num_epochs = 10  # Increased epochs for full training
    batch_size = 32
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for batch in DataLoader(task.train, batch_size=batch_size, shuffle=True):
            loss_info = model.train_step(batch)
            epoch_losses.append(loss_info["loss"])
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    # Evaluation
    print("Evaluating model...")
    model.eval()
    prediction = model.test_step(task.test)
    metrics = task.evaluate(task.test_targets, prediction)
    print("Final Metrics:", metrics)

if __name__ == "__main__":
    main()