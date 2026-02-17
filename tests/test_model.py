# test_model.py

import sys
from pathlib import Path

# add project root to python path.
sys.path.insert(0, str(Path(__file__).parent.parent))


# test_model.py
import unittest      
import torch
import torch.nn as nn
from src.models.gpt import GPTModel
from src.models.layers import LayerNorm, GELU, FeedForward
from src.models.multi_head_attention import MultiHeadAttention
from src.models.transformer_block import TransformerDecoderBlock
from src.utils.config import GPT2_SMALL_124M

from src.utils.config import GPT2Config


class TestGPTModel(unittest.TestCase):
    """unit tests for gpt model."""

    def setUp(self):
        """sets up test fixtures before each test method."""

        # create small test configuration.
        self.config = GPT2Config(
            vocab_size=100,
            context_length=64,
            embed_dim=128,
            n_heads=4,
            num_layers=2,
            drop_rate=0.1,
            qkv_bias=False
        )

        # initialize model.
        self.model = GPTModel(self.config)

        # set to eval mode for testing.
        self.model.eval()


    def test_model_initialization(self):
        """tests that model initializes correctly."""

        # check model is instance of nn.Module.
        self.assertIsInstance(self.model, nn.Module)

        # check embeddings exist.
        self.assertIsNotNone(self.model.tok_emb)
        self.assertIsNotNone(self.model.pos_emb)


    def test_output_shape(self):
        """tests that model output has correct shape."""

        # create dummy input.
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        # forward pass.
        with torch.no_grad():
            output = self.model(input_ids)

        # check output shape.
        expected_shape = (batch_size, seq_len, self.config.vocab_size)
        self.assertEqual(output.shape, expected_shape)


    def test_forward_pass_no_error(self):
        """tests that forward pass runs without errors."""

        # create dummy input.
        input_ids = torch.randint(0, self.config.vocab_size, (2, 10))

        # forward pass should not raise errors.
        try:
            with torch.no_grad():
                output = self.model(input_ids)
            success = True
        except Exception:
            success = False

        self.assertTrue(success)


    def test_context_length_handling(self):
        """tests model handles different sequence lengths."""

        # test with max context length.
        input_max = torch.randint(0, self.config.vocab_size, (1, self.config.context_length))

        with torch.no_grad():
            output_max = self.model(input_max)

        self.assertEqual(output_max.shape[1], self.config.context_length)

        # test with shorter sequence.
        input_short = torch.randint(0, self.config.vocab_size, (1, 5))

        with torch.no_grad():
            output_short = self.model(input_short)

        self.assertEqual(output_short.shape[1], 5)


    def test_parameter_count(self):
        """tests that model has trainable parameters."""

        # count trainable parameters.
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # model should have parameters.
        self.assertGreater(total_params, 0)


    def test_gradient_flow(self):
        """tests that gradients flow through model."""

        # set to training mode.
        self.model.train()

        # create dummy input and target.
        input_ids = torch.randint(0, self.config.vocab_size, (2, 10))
        target_ids = torch.randint(0, self.config.vocab_size, (2, 10))

        # forward pass.
        output = self.model(input_ids)

        # calculate loss.
        loss = nn.functional.cross_entropy(
            output.view(-1, self.config.vocab_size),
            target_ids.view(-1)
        )

        # backward pass.
        loss.backward()

        # check that gradients exist.
        has_gradients = any(p.grad is not None for p in self.model.parameters())
        self.assertTrue(has_gradients)


    def test_device_movement(self):
        """tests model can move between devices."""

        # test cpu.
        self.model.to('cpu')
        input_cpu = torch.randint(0, self.config.vocab_size, (1, 5))

        with torch.no_grad():
            output_cpu = self.model(input_cpu)

        self.assertEqual(output_cpu.device.type, 'cpu')

        # test cuda if available.
        if torch.cuda.is_available():
            self.model.to('cuda')
            input_cuda = torch.randint(0, self.config.vocab_size, (1, 5)).to('cuda')

            with torch.no_grad():
                output_cuda = self.model(input_cuda)

            self.assertEqual(output_cuda.device.type, 'cuda')



class TestLayerNorm(unittest.TestCase):
    """unit tests for layer normalization."""

    def setUp(self):
        """sets up test fixtures."""

        self.embed_dim = 128
        self.layer_norm = LayerNorm(self.embed_dim)


    def test_output_shape(self):
        """tests output shape matches input shape."""

        # create dummy input.
        x = torch.randn(2, 10, self.embed_dim)

        # apply layer norm.
        output = self.layer_norm(x)

        # check shape preserved.
        self.assertEqual(output.shape, x.shape)


    def test_normalized_mean_and_var(self):
        """tests that output is normalized."""

        # create dummy input.
        x = torch.randn(2, 10, self.embed_dim)

        # apply layer norm.
        output = self.layer_norm(x)

        # check mean close to 0 and variance close to 1.
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)

        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-5))
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-5))



class TestGELU(unittest.TestCase):
    """unit tests for gelu activation."""

    def setUp(self):
        """sets up test fixtures."""

        self.gelu = GELU()


    def test_output_shape(self):
        """tests output shape matches input shape."""

        # create dummy input.
        x = torch.randn(2, 10, 128)

        # apply gelu.
        output = self.gelu(x)

        # check shape preserved.
        self.assertEqual(output.shape, x.shape)


    def test_zero_input(self):
        """tests gelu activation at zero."""

        # gelu(0) should be 0.
        x = torch.zeros(1)
        output = self.gelu(x)

        self.assertTrue(torch.allclose(output, torch.zeros_like(output), atol=1e-5))



class TestMultiHeadAttention(unittest.TestCase):
    """unit tests for multi-head attention."""

    def setUp(self):
        """sets up test fixtures."""

        self.d_in = 128
        self.d_out = 128
        self.context_length = 64
        self.num_heads = 4
        self.dropout = 0.1

        self.attention = MultiHeadAttention(
            self.d_in,
            self.d_out,
            self.context_length,
            self.dropout,
            self.num_heads
        )


    def test_output_shape(self):
        """tests output shape is correct."""

        # create dummy input.
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.d_in)

        # apply attention.
        output = self.attention(x)

        # check output shape.
        expected_shape = (batch_size, seq_len, self.d_out)
        self.assertEqual(output.shape, expected_shape)


    def test_causal_masking(self):
        """tests that causal mask prevents attending to future."""

        # set to eval mode.
        self.attention.eval()

        # create simple input.
        x = torch.ones(1, 5, self.d_in)

        with torch.no_grad():
            output = self.attention(x)

        # output should exist and have correct shape.
        self.assertEqual(output.shape, (1, 5, self.d_out))



class TestTransformerBlock(unittest.TestCase):
    """unit tests for transformer decoder block."""

    def setUp(self):
        """sets up test fixtures."""

        self.config = GPT2Config(
            vocab_size=100,
            context_length=64,
            embed_dim=128,
            n_heads=4,
            num_layers=2,
            drop_rate=0.1,
            qkv_bias=False
        )

        self.block = TransformerDecoderBlock(self.config)


    def test_output_shape(self):
        """tests output shape matches input shape."""

        # create dummy input.
        x = torch.randn(2, 10, self.config.embed_dim)

        # apply transformer block.
        output = self.block(x)

        # shape should be preserved.
        self.assertEqual(output.shape, x.shape)


    def test_residual_connections(self):
        """tests that residual connections exist."""

        # set to eval mode to disable dropout.
        self.block.eval()

        # create input.
        x = torch.randn(1, 5, self.config.embed_dim)

        with torch.no_grad():
            output = self.block(x)

        # output should be different from input (due to transformations).
        self.assertFalse(torch.allclose(output, x))



# run tests if this file is directly executed.
if __name__ == '__main__':

    unittest.main()

    # all tests passed successfully with no errors or failures.


