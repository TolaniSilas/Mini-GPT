import torch
from torch import nn


class CausalAttention(nn.Module):
    """causal self-attention with masking to prevent attending to future tokens (preventing the model from peeking or cheating by having context on future tokens)."""

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """initializes causal attention with query, key, value projections and mask."""

        super().__init__()

        # output dimension.
        self.d_out = d_out

        # query projection layer.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)

        # key projection layer.
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)

        # value projection layer.
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # dropout layer.
        self.dropout = nn.Dropout(dropout)

        # causal mask to prevent attending to future positions.
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


    def forward(self, x):
        """applies causal self-attention to input sequence."""

        # get batch size, sequence length, and input dimension.
        batch, num_tokens, d_in = x.shape

        # compute queries, keys, and values.
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # get key dimension for scaling.
        d_k = keys.shape[-1]
        scaling_factor = d_k**0.5

        # compute attention scores.
        attn_scores = queries @ keys.transpose(1, 2)

        # apply causal mask to prevent attending to future tokens.
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        # apply scaled softmax to get attention weights.
        attn_weights = torch.softmax(attn_scores / scaling_factor, dim=-1)

        # apply dropout to attention weights.
        attn_weights = self.dropout(attn_weights)

        # compute weighted sum of values.
        context_vec = attn_weights @ values

        return context_vec
    


class MultiHeadAttention(nn.Module):
    """multi-head attention mechanism with multiple parallel attention heads."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """initializes multi-head attention with multiple causal attention heads."""

        super().__init__()

        # list of attention heads running in parallel.
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for i in range(num_heads)]
        )

    def forward(self, x):
        """applies all attention heads and concatenates their outputs."""

        # run each head and concatenate results along feature dimension.
        return torch.cat([head(x) for head in self.heads], dim=-1)
