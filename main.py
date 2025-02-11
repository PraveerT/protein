from proteinshake.tasks import EnzymeClassTask
from torch_geometric.loader import DataLoader
from model import ComplexEnzymeModel

# Use proteins with Enzyme Class annotations
task = EnzymeClassTask().to_graph(eps=8).pyg()

# Create model
model = ComplexEnzymeModel(task)

# Training using native data loaders
for epoch in range(1):
    for batch in DataLoader(task.train, batch_size=32, shuffle=True):
        loss_info = model.train_step(batch)
        if loss_info["loss"] > 0:
            print(f"Epoch {epoch}, Loss: {loss_info['loss']:.4f}")

# Debug: Print test data structure
# print("\n=== Test Data Structure ===")
# print("Test type:", type(task.test))
# if isinstance(task.test, (list, tuple)):
#     print("Test length:", len(task.test))
#     for i, item in enumerate(task.test):
#         print(f"Item {i} type:", type(item))

# Evaluation with the provided metrics
prediction = model.test_step(task.test)
metrics = task.evaluate(task.test_targets, prediction)

print("\nFinal Metrics:", metrics)