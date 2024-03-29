import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        value_feature, key_feature, query_feature = values.shape[2], keys.shape[2], query.shape[2]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class AttentionFreeTransformer(nn.Module):
    def __init__(self, dim, seq_len, heads=8):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.w_aft = nn.parameter.Parameter(torch.Tensor(heads, seq_len, seq_len))

        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        N, sequence_length, _ = x.size()

        queries = self.query(x).view(N, sequence_length, self.heads, self.head_dim)
        keys = self.key(x).view(N, sequence_length, self.heads, self.head_dim)
        values = self.value(x).view(N, sequence_length, self.heads, self.head_dim)

        aft_values = torch.zeros_like(values)
        for h in range(self.heads):
            for t in range(self.seq_len):
                Z_aft = 0
                o_aft = torch.zeros((N, self.head_dim)).to(x.device)
                for i in range(t+1):
                    att_aft = (self.w_aft[h, t, i] + keys[:, i, h, :]).exp()
                    o_aft += att_aft * values[:, i, h, :]
                    Z_aft += att_aft
                aft_values[:, t, h, :] = o_aft / Z_aft

        aft_values = aft_values.view(N, sequence_length, -1)
        out = self.fc_out(aft_values)

        return out

class RWKVAttention_v1(nn.Module):
    def __init__(self, dim, seq_len, heads=8):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.w_rwkv = nn.parameter.Parameter(torch.Tensor(heads, self.head_dim))
        self.u_rwkv = nn.parameter.Parameter(torch.Tensor(heads, self.head_dim))

        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        N, sequence_length, _ = x.size()

        queries = self.query(x).view(N, sequence_length, self.heads, self.head_dim)
        keys = self.key(x).view(N, sequence_length, self.heads, self.head_dim)
        values = self.value(x).view(N, sequence_length, self.heads, self.head_dim)

        O_rwkv = torch.zeros_like(values)
        for h in range(self.heads):
            a = (keys[:, 0, h, :]).exp() * values[:, 0, h, :]
            b = (keys[:, 0, h, :]).exp()
            for t in range(self.seq_len):
                O_rwkv[:,t, h, :] = (a + torch.exp(self.u_rwkv[h] + keys[:, t, h, :]) * values[:, t, h, :]) / (b + torch.exp(self.u_rwkv[h] + keys[:, t, h, :]))
                a = torch.exp(-self.w_rwkv[h]) * a + torch.exp(keys[:, t, h, :]) * values[:, t, h, :]
                b = torch.exp(-self.w_rwkv[h]) * b + torch.exp(keys[:, t, h, :])

        O_rwkv = O_rwkv.view(N, sequence_length, -1)
        out = self.fc_out(O_rwkv)

        return out

class RWKVAttention_v2(nn.Module):
    """
    增加考虑数值溢出
    """
    def __init__(self, dim, seq_len, heads=8):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.w_rwkv = nn.parameter.Parameter(torch.Tensor(heads, self.head_dim))
        self.u_rwkv = nn.parameter.Parameter(torch.Tensor(heads, self.head_dim))

        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        N, sequence_length, _ = x.size()

        queries = self.query(x).view(N, sequence_length, self.heads, self.head_dim)
        keys = self.key(x).view(N, sequence_length, self.heads, self.head_dim)
        values = self.value(x).view(N, sequence_length, self.heads, self.head_dim)

        O_rwkv = torch.zeros_like(values)
        for h in range(self.heads):
            pp = keys[:, 0, h, :]
            aa = values[:, 0, h, :]
            bb = 1
            O_rwkv[:, 0, h, :] = values[:, 0, h, :]
            for t in range(1, self.seq_len):
                ww = self.u_rwkv[h] + keys[:, t, h, :]
                qq = torch.max(ww, pp)
                e1 = torch.exp(pp - qq)
                e2 = torch.exp(ww - qq)
                a = e1 * aa + e2 * values[:, t, h, :]
                b = e1 * bb + e2
                O_rwkv[:, t, h, :] = a / b
                ww = pp - self.w_rwkv[h]
                qq = torch.maximum(ww, keys[:, t, h, :])
                e1 = torch.exp(ww - qq)
                e2 = torch.exp(keys[:, t, h, :] - qq)
                aa = e1 * aa + e2 * values[:, t, h, :]
                bb = e1 * bb + e2
                pp = qq

        O_rwkv = O_rwkv.view(N, sequence_length, -1)
        out = self.fc_out(O_rwkv)

        return out