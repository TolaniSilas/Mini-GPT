# from utils.config import GPT2Config
# from utils.config import GPT2_SMALL_124M 
import torch
from torch import nn
from .transformer_block import TransformerDecoderBlock


class GPTModel(nn.Module):
    """gpt model with transformer decoder blocks."""

    def __init__(self, config):
        """initializes gpt model with embeddings and transformer layers."""

        super().__init__()

        # token embedding layer.
        self.tok_emb = nn.Embedding(config["vocab_size"], config["embed_dim"])

        # positional embedding layer.
        self.pos_emb = nn.Embedding(config["context_length"], config["embed_dim"])

        # dropout for embeddings.
        self.drop_emb = nn.Dropout(config["drop_rate"])

        # stack of transformer decoder blocks.
        self.trf_blocks = nn.Sequential(
            *[TransformerDecoderBlock(config) for i in range(config["num_layers"])]
        )

        # final layer normalization.
        self.final_norm = nn.LayerNorm(config["embed_dim"])

        # output projection to vocabulary.
        self.out_head = nn.Linear(config["embed_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx):
        """forward pass through the gpt model."""

        # get batch size and sequence length.
        batch_size, seq_len = in_idx.shape

        # get token embeddings.
        tok_embeds = self.tok_emb(in_idx)

        # get positional embeddings.
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # combine token and positional embeddings.
        x = tok_embeds + pos_embeds

        # apply dropout to embeddings.
        x = self.drop_emb(x)

        # pass through transformer blocks.
        x = self.trf_blocks(x)

        # apply final normalization.
        x = self.final_norm(x)

        # project to vocabulary size.
        logits = self.out_head(x)

        return logits