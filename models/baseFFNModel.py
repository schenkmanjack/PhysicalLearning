import torch
import torch.nn as nn

class baseFFNModel(nn.Module):
    def __init__(self, dims, nonlinearity, use_bias=True, dropout=False, p_dropout=0.5):
        super(baseFFNModel, self).__init__()
        self.use_bias = use_bias
        self.dims = dims
        self.nonlinearity = nonlinearity
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1], bias=use_bias)
            self.layers.append(layer)
        self.dropout = dropout
        self.p_dropout = p_dropout
        if dropout:
            self.dropout_layer = nn.Dropout(p=p_dropout)

    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.nonlinearity(x)
                if self.training and self.dropout:
                    mask = torch.rand_like(x) < self.p_dropout
                    x = x * mask
                    # x = self.dropout_layer(x)
        return x
    


    

            