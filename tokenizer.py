import torch

file_path = "data/tiny_shakespeare.txt"

try:
    with open(file_path, "r") as file:
        text = file.read()
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)

vocab = sorted(list(set(text)))


# Create a mapping (dictionary) from tokens to integers
stoi = {token: i for i, token in enumerate(vocab)}

# Create a mapping (dictionary) from integers to tokens
itos = {i: token for i, token in enumerate(vocab)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])



# Convert the whole text to a tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)

print(len(vocab))    
print(len(data) == len(text))