import torch
from torch import nn

from attention import SelfAttention, AttentionFreeTransformer, RWKVAttention_v1, RWKVAttention_v2

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size, dropout):
        super(FeedForward, self).__init__()
        self.fc_1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc_2 = nn.Linear(ff_hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, seq_len, heads, ff_hidden_size, dropout):
        super(EncoderLayer, self).__init__()
        # self.attention = SelfAttention(embed_size, heads)
        # self.attention = AttentionFreeTransformer(embed_size, seq_len, heads)
        self.attention = RWKVAttention_v2(embed_size, seq_len, heads)
        self.norm_1 = LayerNorm(embed_size)
        self.norm_2 = LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x)
        x = self.dropout(self.norm_1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm_2(forward + x))
        return out