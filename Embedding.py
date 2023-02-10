import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super(PositionalEmbedding, self).__init__()
        #since the max seq length of the corpus is less than 200, we use position embeddings here
        self.pe = nn.Embedding(256, config.d_model)
        
    def forward(self, tokens):
        #the input is (batch, seq)
        pos_batch = torch.tensor([i for i in range(tokens.size(1))])
        pos_batch = pos_batch.expand(tokens.size(0), tokens.size(1))
        return self.pe(pos_batch)
        
class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size, config):
        super(TokenEmbedding, self).__init__()
        self.token_embd = nn.Embedding(vocab_size, config.d_model)
        
    def forward(self, tokens):
        #tokens (batch, seq)
        return self.token_embd(tokens.long())#*math.sqrt(config.d_model)