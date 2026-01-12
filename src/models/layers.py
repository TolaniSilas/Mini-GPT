import torch
from torch import nn


class FeedForward(nn.Module):
    """position-wise feed-forward network in transformer. the second sub-layer in each transformer block."""

    def __init__(self, config):
        """initializes feed-forward network with two linear layers."""

        super().__init__()

        # save model configuration for future extraction.
        self.config = config

        # two-layer mlp with gelu activation. it expands to 4x embed_dim then projects back.
        self.layers = nn.Sequential(
            nn.Linear(self.config["embed_dim"], 4 * self.config["embed_dim"]),
            nn.GELU(),
            nn.Linear(4 * self.config["embed_dim"], self.config["embed_dim"]),
        )

    def forward(self, x):
        """applies feed-forward transformation to input."""

        return self.layers(x)


"""
GELU Activation Function Implementation from paper 'Gaussian Error Linear Units (GELUs)' by Hendrycks and Gimpel (2016).
https://arxiv.org/abs/1606.08415
"""

class GELU(nn.Module):
    """gaussian error linear unit (gelu) activation function. mathematical implementation of gelu."""

    def __init__(self):
        """initializes gelu activation."""

        super().__init__()

    def forward(self, x):
        """applies gelu activation to input tensor."""

        # compute gelu activation using tanh approximation.
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            )
        )


"""
Layer Normalization Implementation from paper 'Layer Normalization' by Ba, Kiros, and Hinton (2016).
https://arxiv.org/abs/1607.06450
"""

class LayerNorm(nn.Module):
    """layer normalization for stabilizing neural network training."""

    def __init__(self, embed_dim):
        """initializes layer normalization with learnable parameters."""

        super().__init__()

        # small constant to prevent division by zero.
        self.eps = 1e-5

        # learnable scale parameter.
        self.scale = nn.Parameter(torch.ones(embed_dim))

        # learnable shift parameter.
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        """applies layer normalization to input tensor."""

        # calculate mean across last dimension.
        mean = x.mean(dim=-1, keepdim=True)

        # calculate variance across last dimension.
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # normalize input.
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # apply scale and shift.
        return self.scale * norm_x + self.shift