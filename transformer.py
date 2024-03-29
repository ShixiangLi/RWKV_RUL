import torch
from torch import nn

from encoder import EncoderLayer

class OutputLayer(nn.Module):
    def __init__(self, embed_size, seq_len, ff_hidden_size, dropout, num_classes=2):
        super(OutputLayer, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(embed_size * seq_len, ff_hidden_size)
        self.fc_2 = nn.Linear(ff_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(self.flatten(x))))
        x = self.fc_2(x)
        return x

class Bert(nn.Module):
    def __init__(self, embed_size, seq_len, heads, ff_hidden_size, dropout, num_layers, num_classes=2):
        super(Bert, self).__init__()
        self.encoders = nn.ModuleList(
            [EncoderLayer(embed_size, seq_len, heads, ff_hidden_size, dropout) for _ in range(num_layers)]
        )
        self.output = OutputLayer(embed_size, seq_len, ff_hidden_size, dropout, num_classes=num_classes)

    def forward(self, x, mask):
        for encoder in self.encoders:
            x = encoder(x, mask)
        return self.output(x)