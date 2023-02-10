import torch
import torch.nn as nn
from Transformer import TransformerEncoder, TransformerEncoderLayer
from Embedding import PositionalEmbedding, TokenEmbedding


class ClassificationModel(nn.Module):
    def __init__(self, config, vocab_size=None):
        super(ClassificationModel, self).__init__()
        
        self.pos_embd = PositionalEmbedding(config)
        self.token_embd = TokenEmbedding(vocab_size, config)
        # token = torch.tensor([[0,1,2],[2,3,4]])
        # print(self.pos_embd(token).size(), self.pos_embd(token))
        # print(self.token_embd(token).size(), self.token_embd(token))
        
        #in this classification problem, we just need the encoder part, the output connects to the classifier
        #shape of the output: (batch, seq_len, d_model)
        self.encoder = TransformerEncoder(config)
        
        self.classifier = nn.Sequential(nn.Linear(config.d_model, config.dim_classification),
                                        nn.Dropout(config.dropout),
                                        nn.Linear(config.dim_classification, config.num_class))
        self._init_params()
        
    def _init_params(self):
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, batch, embd_mask=None, key_pad_mask=None ):
        #batch->(batch, seq_len)
        embds = self.token_embd(batch) + self.pos_embd(batch)
       
        out_encoder = self.encoder(embds, att_mask = embd_mask, key_pad_mask =key_pad_mask)
        #(batch, seq_len, d_model) -> (batch, d_model)
        out_encoder = torch.sum(out_encoder, dim=1)
        #(batch, d_model)*( d_model, classes)  -> ( batch, classes)
        out = self.classifier(out_encoder)
        return out 
        