import re


class WordTokenizer:
    """tokenizes text into ids and decodes ids back to text. it focuses on word-level tokenization."""

    def __init__(self, vocab):
        """initializes tokenizer with vocabulary mappings."""

        self.tok_to_int = vocab
        self.int_to_tok = {integer: token for token, integer in vocab.items()}
        self.pattern = r'([,.:;?_!"()\'\[\]{}\/\\|—–-]+|\.\.\.|\s+)'


    def encode(self, text):
        """converts text to list of token ids."""

        # split on punctuation and whitespace.
        preprocessed = re.split(self.pattern, text)

        # remove empty strings and whitespace.
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # replace unknown tokens with <|unk|>.
        preprocessed = [token if token in self.tok_to_int else "<|unk|>" for token in preprocessed]

        # convert tokens to ids.
        ids = [self.tok_to_int[tok] for tok in preprocessed]

        # return ids.
        return ids


    def decode(self, ids):
        """converts list of token ids back to text."""

        # map ids to tokens.
        tokens = [self.int_to_tok[id] for id in ids]

        # join tokens with spaces.
        text = " ".join(tokens)

        # remove spaces before punctuation.
        text = re.sub(self.pattern, r'\1', text)

        # remove spaces before punctuation.
        text = re.sub(r'\s+([,.:;?_!"()\'\[\]{}\/\\|—–-])', r'\1', text)

        return text