import wandb
import torch

# 1. Start a new run
wandb.init(project="my-ml-project", name="first-experiment")

# 2. Define model, optimizer, loss
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Dummy data
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# 3. Training loop
for epoch in range(10):
    optimizer.zero_grad()
    preds = model(x)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()

    # 4. Log metrics to WandB
    wandb.log({"epoch": epoch, "loss": loss.item()})

# 5. Save model
torch.save(model.state_dict(), "model.pt")
wandb.save("model.pt")   # Uploads it to WandB
