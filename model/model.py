from numpy.ma import mask_cols
from torch import nn
from tokenizer import vocab
import numpy as np
import torch
import torch.nn.functional as F



# Hyperparameters
vocab_size = len(vocab)
block_size = 64
batch_size = 32
embedding_dim = 128
num_heads = 4
num_layers = 4



# Embedding Size is (65, 128)
embedding_table = nn.Embedding(vocab_size, embedding_dim)


# its block_size by embedding_dim because we want to add the position embedding to the input sentence
# we only care about the position of the tokens in the sentence, not the tokens themselves
position_embedding_table = nn.Embedding(block_size, embedding_dim)




# Since embedding_dim = 128 and num_heads = 4, we need to divide the embedding_dim by num_heads to get the head_dim
# So per head Q, K, V Shape -> (batch_size, block_size, head_dim) aka (32, 64, 32)
head_dim = embedding_dim // num_heads

# Q, K, V are the queries, keys, and values for the attention mechanism
# Q is the query vector, K is the key vector, V is the value vector

#Attention(Q, K, V) = softmax(QK^T / sqrt(head_dim)) V

class AttentionHead(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.W_q = nn.Linear(embedding_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, head_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, head_dim, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)),
        )  # tril is to create a lower triangular matrix


    def forward(self, x):
        B, T, C = x.shape #B is the batch size, T is the sequence length, C is the embedding dimension

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        scores = Q @ K.transpose(-2, -1) / head_dim**0.5 #transpose(-2, -1) is to transpose the last two dimensions of the tensor and **0.5 is another waying sqrt(head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf')) #masked_fill is to fill the scores with -inf where the mask is 0
        weights = torch.softmax(scores, dim=-1) #softmax is to normalize the scores, dim=-1 is to normalize the last dimension
        return weights @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_dim) for _ in range(num_heads)]) #We create a list of attention heads
        self.W_o = nn.Linear(embedding_dim, embedding_dim) #We create a linear layer to project the output of the attention heads back to the embedding dimension

    def forward(self, x):
        return self.W_o(torch.cat([head(x) for head in self.heads], dim=-1)) #We concatenate the output of the attention heads and project it back to the embedding dimension

class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.W_1 = nn.Linear(embedding_dim, 4 * embedding_dim) #We create a linear layer to project the input to the feed forward network
        self.activation = nn.GELU()
        self.W_2 = nn.Linear(4 * embedding_dim, embedding_dim) #We create a linear layer to project the output of the feed forward network back to the embedding dimension

    def forward(self, x):
        return self.W_2(self.activation(self.W_1(x))) #We apply the GELU activation function to the output of the feed forward network and project it back to the embedding dimension
                                                 #GELU is a type of activation function that allows for non-linearity so we can have more complex patterns in the data

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads)
        self.ff = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embedding_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None): #idx is the input tensor of shape (batch_size, sequence_length) each value is a integer representing a token id
        B, T = idx.shape

        tok_embs = self.token_embedding_table(idx)
        pos_embs = self.position_embedding_table(torch.arange(T, device=idx.device))

        x = tok_embs + pos_embs
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
            