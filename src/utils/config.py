 
class GPT2Config:
    """configuration class for gpt-2 model hyperparameters."""

    def __init__(
        self,
        vocab_size=50257,   # the model's vocabulary size
        context_length=1024,  # the model's context length
        emb_dim=768,    # the embedding function
        n_heads=12,    # the number of attention heads
        n_layers=12,   # the number of transformer layers
        drop_rate=0.2,  # the probability of drop out rate.
        qkv_bias=False   # the query-key-value bias.
    ):
        """initializes gpt-2 configuration with hyperparameters."""

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias
                 


# predefined configurations for different gpt-2 variants.
GPT2_SMALL_124M = GPT2Config(
    vocab_size=50257,
    context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate=0.1,
    qkv_bias=False
)

GPT2_MEDIUM = GPT2Config(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1024,
    n_heads=16,
    n_layers=24,
    drop_rate=0.1,
    qkv_bias=False
)

GPT2_LARGE = GPT2Config(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1280,
    n_heads=20,
    n_layers=36,
    drop_rate=0.1,
    qkv_bias=False
)

GPT2_XL = GPT2Config(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1600,
    n_heads=25,
    n_layers=48,
    drop_rate=0.1,
    qkv_bias=False
)