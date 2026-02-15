from torch import nn
from .multi_head_attention import MultiHeadAttention
from .layers import FeedForward


class TransformerDecoderBlock(nn.Module):
    """transformer decoder block with multi-head attention and feed-forward layers."""

    def __init__(self, config):
        """initializes transformer block with attention and feed-forward components."""

        super().__init__()

        # multi-head attention layer.
        self.multi_att = MultiHeadAttention(
            d_in=config["embed_dim"],
            d_out=config["embed_dim"],
            context_length=config["context_length"],
            num_heads=config["n_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"]
        )

        # layer normalization layers.
        self.norm1 = nn.LayerNorm(config["embed_dim"])
        self.norm2 = nn.LayerNorm(config["embed_dim"])

        # dropout for residual connections.
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

        # feed-forward network.
        self.feed_for = FeedForward(config)

    def forward(self, x):
        """applies transformer block with residual connections."""

        ## -------  shortcut connection for multi-head attention sub-layer.  -------

        # store the original input.
        shortcut = x

        # apply layer normalization before the multi-head attention sub-layer.
        x = self.norm1(x)

        # apply the output of the layer normalization to the multi-head attention sub-layer.
        x = self.multi_att(x)

        # apply drop out.
        x = self.drop_shortcut(x)

        # add the original input back.
        x = x + shortcut

        ## ------  shortcut connection for feed forward sub-layer.  ------

        # store the original input.
        shortcut = x

        # apply layer normalization before the feed forward sub-layer.
        x = self.norm2(x)

        # apply the output of the layer normalization to the feed forward sub-layer.
        x = self.feed_for(x)

        # apply drop out.
        x = self.drop_shortcut(x)

        # add the original input back.
        x = x + shortcut

        return x


