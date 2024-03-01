import torch
from torch import nn

from models.layers.multihead_attention import MultiHeadAttention
from models.layers.normalization import Normalization
from models.layers.feed_forward import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderBlock, self).__init__()
        self.attention_1 = MultiHeadAttention(d_model, n_head)
        self.dropout_1 = nn.Dropout(p=drop_prob)
        self.norm_1 = Normalization(d_model)
        self.attention_2 = MultiHeadAttention(d_model, n_head)
        self.dropout_2 = nn.Dropout(p=drop_prob)
        self.norm_2 = Normalization(d_model)
        self.feed_forward = FeedForward(d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.dropout_3 = nn.Dropout(p=drop_prob)
        self.norm_3 = Normalization(d_model)
    
    def forward(self, dec, enc, src_mask, trg_mask):
        _x = dec
        x = self.attention_1(q=dec, k=dec, v=dec, mask=trg_mask)
        x = self.dropout_1(x)
        x = self.norm_1(x + _x)

        if enc is not None:
            _x = x
            x = self.attention_2(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout_2(x)
            x = self.norm_2(x + _x)
        
        _x = x
        x = self.feed_forward(x)
        x = self.dropout_3(x)
        x = self.norm_3(x + _x)

        return x