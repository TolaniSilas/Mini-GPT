import tiktoken


class BPETokenizer:
    """tokenizes text using byte pair encoding (bpe). uses tiktoken's gpt-2 tokenizer."""

    def __init__(self, model_name="gpt2"):
        """initializes bpe tokenizer with specified model."""

        # load tiktoken encoder for the specified model.
        self.tokenizer = tiktoken.get_encoding(model_name)

        # store vocab size.
        self.vocab_size = self.tokenizer.n_vocab

    def encode(self, text, allowed_special=None):
        """converts text to list of token ids."""

        # set default allowed special tokens.
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}

        # convert text to token ids.
        ids = self.tokenizer.encode(text, allowed_special=allowed_special)

        return ids

    def decode(self, ids):
        """converts list of token ids back to text."""

        # convert token ids back to text.
        text = self.tokenizer.decode(ids)

        return text

    def get_vocab_size(self):
        """returns the vocabulary size of the tokenizer."""

        return self.vocab_size