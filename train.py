import torch
from model.model import GPT
from dataset import get_batch
print("Starting training...")
# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Hyperparameters
# -----------------------------
max_iters = 2000
eval_interval = 1000
learning_rate = 3e-4

# -----------------------------
# Model
# -----------------------------
model = GPT().to(device)
model.load_state_dict(torch.load("tiny_gpt.pt", map_location=device), strict=False)
print("Loaded pretrained weights.")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Model parameters:", sum(p.numel() for p in model.parameters()))

# -----------------------------
# Training loop
# -----------------------------
for step in range(max_iters):

    # get training batch
    x, y = get_batch()
    x = x.to(device)
    y = y.to(device)

    # forward
    logits, loss = model(x, y)

    # backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # logging
    if step % eval_interval == 0:
        print(f"step {step} | loss {loss.item():.4f}")

# -----------------------------
# Save model
# -----------------------------
torch.save(model.state_dict(), "tiny_gpt.pt")

print("Training finished and model saved.")