import unittest
from src.tokenizers import WordTokenizer
from src.tokenizers import BPETokenizer


class TestWordTokenizer(unittest.TestCase):
    """unit tests for word tokenizer."""

    def setUp(self):
        """sets up test fixtures before each test method."""

        # create simple and sample vocabulary.
        self.vocab = {
            "hello": 0,
            "world": 1,
            "test": 2,
            "this": 3,
            "is": 4,
            "a": 5,
            ",": 6,
            ".": 7,
            "!": 8,
            "<|unk|>": 9,
            "<|endoftext|>": 10
        }

        # initialize word tokenizer.
        self.tokenizer = WordTokenizer(self.vocab)


    def test_encode_known_tokens(self):
        """tests encoding of known tokens."""

        # test basic encoding.
        text = "hello world"
        expected_ids = [0, 1]

        self.assertEqual(self.tokenizer.encode(text), expected_ids)


    def test_encode_with_punctuation(self):
        """tests encoding with punctuation marks."""

        # test various punctuation handling.
        text = "hello, world!"
        expected_ids = [0, 3, 1, 5]

        self.assertEqual(self.tokenizer.encode(text), expected_ids)


    def test_encode_unknown_tokens(self):
        """tests encoding of unknown tokens."""

        # test unknown token handling.
        text = "hello <|unk|>"
        expected_ids = [0, 6]  

        self.assertEqual(self.tokenizer.encode(text), expected_ids)


    def test_decode_basic(self):
        """tests basic decoding of token ids."""

        # test basic decoding.
        ids = [0, 1]
        expected_text = "hello world"

        self.assertEqual(self.tokenizer.decode(ids), expected_text)


    def test_decode_with_punctuation(self):
        """tests decoding with punctuation spacing."""

        # test punctuation spacing.
        ids = [0, 3, 1, 5]
        expected_text = "hello, world!"

        self.assertEqual(self.tokenizer.decode(ids), expected_text)


    def test_encode_decode_roundtrip(self):
        """tests that encode then decode returns similar text."""

        # test roundtrip.
        text = "hello, world! this is a test."

        # encode the sample text.
        encoded = self.tokenizer.encode(text)

        # decode the encoded sample text
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(decoded.strip(), text.strip())



class TestBPETokenizer(unittest.TestCase):
    """unit tests for byte-pair encoding (bpe) tokenizer."""

    def setUp(self):
        """sets up test fixtures before each test method."""

        # initialize bpe tokenizer.
        self.tokenizer = BPETokenizer(model_name="gpt2")


    def test_encode_basic(self):
        """tests basic encoding."""

        # test basic encoding.
        text = "hello world"
        ids = self.tokenizer.encode(text)

        self.assertIsInstance(ids, list)
        self.assertTrue(len(ids) > 0)


    def test_encode_with_special_tokens(self):
        """tests encoding with special tokens."""

        # test special token handling.
        text = "hello <|endoftext|> world"
        ids = self.tokenizer.encode(text)

        self.assertIsInstance(ids, list)


    def test_decode_basic(self):
        """tests basic decoding."""

        # test decoding.
        text = "hello world"

        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)

        self.assertEqual(decoded, text)


    def test_vocab_size(self):
        """tests vocabulary size retrieval."""

        # test vocab size.
        vocab_size = self.tokenizer.get_vocab_size()  # get the vocabulary size.

        self.assertEqual(vocab_size, 50257)


    def test_encode_decode_roundtrip(self):
        """tests that encode then decode preserves text."""

        # test roundtrip.
        text = "the quick brown fox jumps over the lazy dog. this is a method for bpe tokenizer text."

        # decode the encoded sample text.
        ids = self.tokenizer.encode(text)
        
        # decode the encoded sample text.
        decoded = self.tokenizer.decode(ids)
        
        self.assertEqual(decoded, text)


    def test_empty_string(self):
        """tests encoding and decoding empty strings."""

        # test empty string.
        text = ""

        ids = self.tokenizer.encode(text)
        self.assertEqual(ids, [])

        decoded = self.tokenizer.decode(ids)
        self.assertEqual(decoded, "")



# run tests if this file is directly executed.
if __name__ == '__main__':
    unittest.main()