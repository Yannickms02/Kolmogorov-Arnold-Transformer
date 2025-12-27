import torch.nn as nn

class StandardFFN(nn.Module):
    """
    A1: Deep Standard FFN
    Structure: Linear(in) -> [Linear(inner) * (n-1)] -> Linear(out)
    """

    def __init__(self, d_model, d_ff, n_hidden=0, dropout=0.1):
        super().__init__()

        layers = []

        # 1. Input Projection (First layer: d_model -> d_ff)
        layers.append(nn.Linear(d_model, d_ff))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        # 2. Additional depth (Hidden layers: d_ff -> d_ff)
        # Skipped when n_hidden=0 (default transformer FFN)
        for _ in range(n_hidden):
            layers.append(nn.Linear(d_ff, d_ff))
            layers.append(nn.LayerNorm(d_ff)) # LN added for deeper FFN blocks
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        # 3. Output Projection (Final layer: d_ff -> d_model)
        layers.append(nn.Linear(d_ff, d_model))

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        # x is passed through all layers of sequential block
        return self.net(x)