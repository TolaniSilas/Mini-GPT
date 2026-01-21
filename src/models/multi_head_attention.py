import torch
from torch import nn



class MultiHeadAttention(nn.Module):
    """multi-head attention mechanism with multiple parallel attention heads."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """initializes multi-head attention with projections and causal mask."""

        super().__init__()

        # ensure output dimension is divisible by number of heads.
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # store dimensions.
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # query projection layer.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)

        # key projection layer.
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)

        # value projection layer.
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # output projection to combine heads.
        self.out_proj = nn.Linear(d_out, d_out)

        # dropout layer.
        self.dropout = nn.Dropout(dropout)

        # causal mask to prevent attending to future positions (in order to avoid cheating).
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


    def forward(self, x):
        """applies multi-head attention to input sequence."""

        # get batch size, sequence length, and input dimension.
        b, num_tokens, d_in = x.shape

        # compute queries, keys, and values.
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # split into multiple heads by reshaping.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # rearrange to (batch, num_heads, seq_len, head_dim).
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # compute attention scores.
        attn_scores = queries @ keys.transpose(2, 3)

        # apply causal mask to prevent attending to future tokens.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # get key dimension for scaling.
        d_k = keys.shape[-1]
        scaling_factor = d_k**0.5

        # apply scaled softmax to get attention weights.
        attn_weights = torch.softmax(attn_scores / scaling_factor, dim=-1)

        # apply dropout to attention weights.
        attn_weights = self.dropout(attn_weights)

        # compute weighted sum of values.
        context_vec = attn_weights @ values

        # rearrange back to (batch, seq_len, num_heads, head_dim).
        context_vec = context_vec.transpose(1, 2)

        # combine all heads by concatenating.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # apply output projection.
        context_vec = self.out_proj(context_vec)

        return context_vec