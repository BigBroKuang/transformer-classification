from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self._init_params()
        
    def _init_params(self):
        for p in self.parameters():
            if p.dim()>0:
                xavier_uniform_(p)
                
    def forward(self, embds, out_embds, src_mask=None, out_mask=None, memory_mask=None, src_key_pad_mask=None,
                out_key_pad_mask=None, memory_key_pad_mask=None ):
        #encoder -> 
        memory = self.encoder(embds, att_mask =src_mask, key_pad_mask=src_key_pad_mask)
        output = self.decoder(out_embds, memory, out_mask=out_mask, memory_mask=memory_mask, memory_key_pad_mask=memory_key_pad_mask, out_key_pad_mask=out_key_pad_mask)
        return output

        
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.num_layers = config.num_encoder_layers
        
    def forward(self, embds, att_mask = None, key_pad_mask =None):
        output = embds
        for encoder in self.layers:
            output = encoder(output, att_mask = att_mask, key_pad_mask =key_pad_mask)
        return output 
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()
        self.atten = MultiheadAttention(config)
        
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.activation = F.relu
        self.norm2 = nn.LayerNorm(config.d_model)
        
    def forward(self, sum_embds, att_mask=None, key_pad_mask=None):
        attout = self.atten(sum_embds, sum_embds, sum_embds, att_mask, key_pad_mask)
        #ResNet
        sum_embds = self.norm1(sum_embds + self.dropout1(attout))
        #FF, ResNet
        attout = self.activation(self.linear1(sum_embds))
        attout = self.linear2(self.dropout2(attout)) 
        sum_embds =self.norm2(sum_embds + self.dropout3(attout))
        return sum_embds

     
class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.num_layers = config.num_decoder_layers
        
    def forward(self, out_embds, memory, out_mask=None, memory_mask=None, memory_key_pad_mask=None, out_key_pad_mask=None):
        output = out_embds
        for decoder in self.layers:
            output = decoder(output, memory, out_mask=out_mask, memory_mask=memory_mask, memory_key_pad_mask=memory_key_pad_mask, out_key_pad_mask=out_key_pad_mask)
        return output 
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerDecoderLayer, self).__init__()
        self.self_atten = MultiheadAttention(config)
        self.atten = MultiheadAttention(config)
        
        
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.dropout4 = nn.Dropout(config.dropout)
        
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        
        self.activation = F.relu
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        
    def forward(self, out_embds, memory, out_mask=None, memory_mask=None, memory_key_pad_mask=None, out_key_pad_mask=None):
        #output self attention
        self_att_out = self.self_atten(out_embds, out_embds, out_embds, att_mask =out_mask, key_pad_mask=out_key_pad_mask)
        self_att_out = self.norm1(out_embds + self.dropout1(self_att_out))
        #output 
        att_out = self.atten(self_att_out,memory,memory,att_mask=memory_mask, key_pad_mask=memory_key_pad_mask)
        self_att_out = self.norm2(self_att_out + self.dropout2(att_out))
        
        att_out = self.activation(self.linear1(self_att_out))
        att_out = self.linear2(self.dropout3(att_out)) 

        att_out =self.norm3(self_att_out + self.dropout4(att_out))
        return att_out
    
class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super(MultiheadAttention, self).__init__()
        
        self.d_model = config.d_model
        self.head_dim = config.d_model//config.num_head
        self.num_head = config.num_head
        self.dropout = config.dropout
        
        self.q = nn.Linear(config.d_model, config.d_model)
        self.k = nn.Linear(config.d_model, config.d_model)
        self.v = nn.Linear(config.d_model, config.d_model)
        self.att_out = nn.Linear(config.d_model, config.d_model)
        
        self._reset_params()
    def _reset_params(self):
        for v in self.parameters():
            if v.dim()>1:
                xavier_uniform_(v)
    def forward(self, query, key, value, att_mask=None, key_pad_mask=None):
        #query,key,val -> (batch, seq_len, d_model)
        #att_mask -> (q_len, k_len)
        #key_pad_mask -> (batch, q_len)
        btsz, q_len, d_model = query.size()
        k_len = key.size(1)
        
        q_cal = self.q(query)
        k_cal = self.k(key)
        v_cal = self.v(value)
        
        scaling_factor = float(self.head_dim)**(-0.5)
        q_cal = q_cal*scaling_factor

        
        #expand the dimension of batch
        #(batch, seq_len, dim)->(seq_len, batch, dim) -> (seq_len, batch*num_head, head_dim) ->  (batch*num_head, seq_len, head_dim)
        q_cal = q_cal.transpose(0,1).reshape(-1, btsz*self.num_head, self.head_dim).transpose(0, 1)
        k_cal = k_cal.transpose(0,1).reshape(-1, btsz*self.num_head, self.head_dim) .transpose(0, 1)
        v_cal = v_cal.transpose(0,1).reshape(-1, btsz*self.num_head, self.head_dim) .transpose(0, 1)
        #(batch*num_head, q_len, head_dim) *(batch*num_head, k_len, head_dim)->(batch*num_head, q_len, k_len)
        atten_output = torch.bmm(q_cal, k_cal.transpose(1, 2))
        
        if att_mask is not None:
            #if att_mask not specified, uncomment the following syntax
            #att_temp = torch.ones(q_len, k_len)
            #att_temp = torch.tril(att_temp)
            #att_mask = torch.where(att_temp>0, 0, float('-inf'))
            #print(atten_output.size(), att_mask.size())
            att_mask = att_mask.unsqueeze(0) #may have mistakes when the dimensions are the same, expand the dimension can resolve the problem
            atten_output += att_mask #broadcast add
            
        if key_pad_mask is not None:
            #in decoder, query is predicted from k, v
            # expand the dimension, (batch*num_head, q_len, k_len) -> (batch, num_head, q_len, k_len)
            atten_output = atten_output.reshape(btsz, self.num_head, q_len, k_len) 
            #(btsz, 1, 1, k_len),k can be regarded as future words
            atten_output = atten_output.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), -np.inf)
            #(batch, num_head, q_len, k_len) -> (batch*num_head, q_len, k_len)
            atten_output = atten_output.reshape(btsz*self.num_head, q_len, k_len)
            
        atten_output = F.softmax(atten_output, dim=-1)
        atten_output = F.dropout(atten_output, p=self.dropout, training = self.training)
        #(batch*num_head, q_len, k_len) *(batch*num_head, k_len, head_dim) -> (batch*num_head, q_len, head_dim)
        atten_output = torch.bmm(atten_output, v_cal)
        # (batch*num_head, q_len, head_dim)->  ( q_len, batch*num_head, head_dim)->( q_len, batch, d_model) -> ( batch,  q_len, embd)
        atten_output = atten_output.transpose(0,1).reshape(q_len, btsz, self.d_model).transpose(0, 1)
        #z = self.att_out(atten_output)
        return self.att_out(atten_output)
        
        
        
            
