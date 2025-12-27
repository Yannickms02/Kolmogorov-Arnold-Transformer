import torch
from torch import nn
import torch.nn.functional as F

from models.layers.ffn_grkan import GRKANFFN
from models.layers.ffn_mlp import StandardFFN
from models.layers.ffn_kan import KANFFN
from models.layers.ffn_mlp_bspline import BSplineFFN


class CustomTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_type="mlp", d_ff=None, n_hidden=1, dropout=0.1, is_causal=False):
        super().__init__()

        # 1. Self-Attention Part (Standard PyTorch)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Feed Forward Part (Modular)
        self.norm2 = nn.LayerNorm(d_model)

        if d_ff is None:
            d_ff = 4 * d_model

        # Swappable FFN-block
        if ffn_type == "mlp":         # A1
            self.ffn = StandardFFN(d_model, d_ff, n_hidden, dropout)
        elif ffn_type == "kan_bspline": # A2
            self.ffn = KANFFN(d_model, d_ff, aggregation='sum', n_hidden=n_hidden)
        elif ffn_type == "kan_mean": # A3
            self.ffn = KANFFN(d_model, d_ff, aggregation='mean', n_hidden=n_hidden)
        elif ffn_type == "kan_grkan":      # A4
            self.ffn = GRKANFFN(
                d_model=d_model,
                d_ff=d_ff,
                n_groups=8,        # Standard KAT: 8 Groups
                num_degree=5,        # Numerator: a0 to a5 (6 coefficients)
                denom_degree=4,      # Denominator: b1 to b4 (4 coefficients)
                dropout=dropout,
                n_hidden=n_hidden,
                use_layernorm=True
            )
        elif ffn_type == "mlp_bspline": # A5
            self.ffn = BSplineFFN(
                d_model=d_model,
                d_ff=d_ff,
                n_hidden=n_hidden,
                dropout=dropout,
                grid_size=5,
                grid_range=(-1,1),
                spline_order=3
            )
        else:
            raise ValueError(f"Invalid FFN type: {ffn_type}")

        self.dropout2 = nn.Dropout(dropout)
        self.is_causal = is_causal

    def forward(self, x, attn_mask=None):
        # Pre-LN Transformer
        # 1. Attention Block
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False, attn_mask=attn_mask, is_causal=self.is_causal)
        x = x + self.dropout1(attn_out) # Skip Connection

        # 2. FFN Block
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out) # Skip Connection

        return x



class KATClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=256, n_heads=4,
                 n_layers=4, n_hidden=0, ffn_type="mlp", d_ff=1024, dropout=0.1, max_seq_len=512):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Embedding & Positional Encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Stacked Transformer Blocks
        self.layers = nn.ModuleList([
            CustomTransformerBlock(d_model, n_heads, ffn_type=ffn_type, d_ff=d_ff, n_hidden=n_hidden, dropout=dropout, is_causal=False)
            for _ in range(n_layers)
        ])

        # Final classification head
        self.norm_final = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x: (batch, seq_len) Indices
        b, seq_len = x.shape
        device = x.device

        # Position Indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        # Embedding with indices
        x = self.drop(self.embedding(x) + self.pos_embedding(positions))

        # Propagate through all layers
        for layer in self.layers:
            x = layer(x)

        # Global Average Pooling für Classification
        x = self.norm_final(x)
        x = x.mean(dim=1)  # (batch, d_model)

        # Classify
        logits = self.classifier(x)
        return logits

class KATLanguageModelling(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4,
                 n_layers=4, n_hidden=0, ffn_type="mlp", d_ff=1024, dropout=0.1, max_seq_len=512):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Embedding & Positional Encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Stacked Transformer Blocks
        self.layers = nn.ModuleList([
            CustomTransformerBlock(d_model, n_heads, ffn_type=ffn_type, d_ff=d_ff, n_hidden=n_hidden, dropout=dropout, is_causal=True)
            for _ in range(n_layers)
        ])

        # Final classification head
        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Weight Tying
        self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: (batch, seq_len)
            targets: (batch, seq_len) - training
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar if targets given
        """
        b, seq_len = input_ids.shape
        device = input_ids.device

        # Causal Mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        # Position Indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        # Embeddings
        x = self.drop(self.embedding(input_ids) + self.pos_embedding(positions))

        # Masked Self-Attention
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask)

        x = self.norm_final(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for CrossEntropy: (Batch*Seq, Vocab) vs (Batch*Seq)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100  # Padding
            )

        return logits, loss