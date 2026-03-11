import torch
from model.model import GPT
from tokenizer import encode, decode

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Load trained model
# -----------------------------
model = GPT().to(device)
model.load_state_dict(torch.load("tiny_gpt.pt", map_location=device), strict=False)
model.eval()

# -----------------------------
# Text generation function
# -----------------------------
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=20):

    for _ in range(max_new_tokens):

        # keep only last block_size tokens
        idx_cond = idx[:, -model.position_embedding_table.num_embeddings:]

        # forward pass
        logits, _ = model(idx_cond)

        # take logits from last token
        logits = logits[:, -1, :]

        # temperature scaling
        logits = logits / temperature

        # top-k filtering
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float('-inf')

        # convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        # append token
        idx = torch.cat((idx, next_token), dim=1)

    return idx


# -----------------------------
# Prompt
# -----------------------------
prompt = "ROMEO:"
idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

# -----------------------------
# Generate text
# -----------------------------
output = generate(model, idx, max_new_tokens=200)

# -----------------------------
# Decode and print
# -----------------------------
text = decode(output[0].tolist())
print("\nGenerated text:\n")
print(text)