import torch
from tokenizer import data

# Hyperparameters
block_size = 128
batch_size = 128
embedding_dim = 512
num_heads = 8
num_layers = 8


def get_batch():
    # Randomly select a starting index for the batch
    # len(data) - block_size is the maximum starting index
    # (batch_size,) is the shape of the tensor
    # torch.randint() is used to generate random integers
    # The random integers are between 0 and len(data) - block_size
    # The random integers are of shape (batch_size,)
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    
    # Create the input and output tensors
    # torch.stack() is used to stack the tensors along a new dimension
    # The new dimension is the batch dimension
    # The input tensor is the data from the starting index to the starting index + block_size
    # The output tensor is the data from the starting index + 1 to the starting index + block_size + 1
    x = torch.stack([data[i:i+block_size] for i in ix]) # (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # (batch_size, block_size)
    
    return x, y
