# Mini-GPT

## Project Overview
This repository implements the **Generative Pretrained Transformer (GPT)** series (e.g., GPT-2, GPT-3, and later variants). The model architecture is the foundational building block underlying the GPT family, and its design is inspired by a major breakthrough in deep learning: the **self-attention mechanism** introduced in “Attention Is All You Need” by Vaswani et al. 

The architecture utilizes a **decoder-only Transformer** design and incorporates robust text-preprocessing techniques, such as **Byte Pair Encoding (BPE)** tokenization. BPE enables the model to effectively handle words that were not present in the training dataset by decomposing them into subword units, thereby improving vocabulary coverage and generalization.

**TL;DR:** This repository includes the coding and implementation from scratch of the following papers: [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [*Language Models are Few-Shot Learners*](https://arxiv.org/pdf/2005.14165), [*Attention Is All You Need*](https://arxiv.org/pdf/1706.03762) and so on (Check references for full details).

The models in this repository were trained on **Outliers book** by Malcolm Gladwell solely for experimental purposes. This experimentation was motivated by my curiosity and interaction of ideas from my course work, **ECE 518: Neural Networks**.


---

## Installation

This project uses `uv` for fast and efficient dependency management. All code were written or implemented on Linux machine.

### Prerequisites
- Python >= 3.10
- `uv` package installer

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on macOS/Linux:
```bash
pip install uv
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/TolaniSilas/Mini-GPT.git
cd Mini-GPT
```

2. Create and activate virtual environment:
```bash
uv venv

# On macOS and Linux, use:
source .venv/bin/activate  

# While on Windows, use: 
.venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

Or install with development dependencies:
```bash
uv pip install -e ".[dev]"
```

### CPU-Only PyTorch Installation

If you want CPU-only PyTorch (faster installation, smaller size):
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv add "tiktoken>=0.5.0" "pypdf>=3.0.0" "matplotlib>=3.7.0"
```

### Verify Installation
```bash
python -c "import torch; import tiktoken; print('installation completed or successful!')"
```

---

## Usage

### 1. Preprocess Data

Convert PDF files to preprocessed text:
```bash
python3 scripts/preprocess_data.py
```
This preprocessed text will be utilized to train the language model.

**Arguments:**
- `--raw_dir`: Directory containing raw PDF files (default: `data/raw`)
- `--output_dir`: Output directory for processed text files (default: `data/processed`)

### 2. Train Model

Train the GPT-2 model on your preprocessed data:
```bash
python3 scripts/train.py
```

**Arguments:**
- `--data_path`: Path to processed data (default: `data/processed`).
- `--checkpoint_dir`: Directory to save checkpoints (default: `results/checkpoints`).
- `--batch_size`: Batch size for training (default: `8`).
- `--epochs`: Number of training epochs (default: `15`).
- `--lr`: Learning rate (default: `0.0004`).
- `--weight_decay`: Weight decay for regularization (default: `0.1`).
- `--device`: Device to use - `cuda` or `cpu` (default: auto-detect).

**Training Tips:**
- Start with `--batch_size 2` or `4` if you have limited RAM.
- Use `--epochs 1` for quick testing.
- Monitor training progress in the console output.

### 3. Generate Text

Generate text using a trained model:
```bash
python3 scripts/generate.py
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint (required)
- `--prompt`: Starting text prompt (default: `"The meaning of life is"`)
- `--max_tokens`: Maximum tokens to generate (default: `100`)
- `--temperature`: Sampling temperature (default: `1.0`)
- `--device`: Device to use - `cuda` or `cpu` (default: auto-detect)

---


## Project Structure
```
Mini-GPT/
├── data/
│   ├── raw/                 # original pdf files
│   ├── processed/           # preprocessed text files
├── notebooks/
│   ├── model_architecture.ipynb
│   ├── tokenization.ipynb
│   └── training.ipynb
├── papers/                  # reference papers
├── results/
│   └── plots/              # training visualizations
├── scripts/
│   ├── generate.py         # text generation
│   ├── helpers_functions.py # utility functions
│   ├── preprocess_data.py  # data preprocessing
│   └── train.py            # training script
├── src/
│   ├── data/               # data processing utilities
│   ├── models/             # model architecture
│   ├── tokenizers/         # tokenization implementations
│   ├── utils/              # helper functions and configs
│   └── __init__.py
├── tests/
│   ├── test_model.py
│   └── test_tokenizers.py
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

---

## Model Configurations

The repository includes several GPT model configurations defined in `src/utils/config.py`:

### GPT2-Small (Custom - ~134M params)
```python
vocab_size: 50257
context_length: 256
embed_dim: 252  # reduced from 768 due to computational limitations
n_heads: 12
num_layers: 12
drop_rate: 0.2
```

### GPT2-Medium (355M params)
```python
vocab_size: 50257
context_length: 1024
embed_dim: 1024
n_heads: 16
num_layers: 24
drop_rate: 0.2
```

### GPT2-Large (774M params)
```python
vocab_size: 50257
context_length: 1024
embed_dim: 1280
n_heads: 20
num_layers: 36
drop_rate: 0.2
```

### GPT3 (175B params - specification only)
```python
vocab_size: 50257
context_length: 2048
embed_dim: 12288
n_heads: 96
num_layers: 96
drop_rate: 0.2
```

**Note:** This implementation uses only the GPT2-Small configuration due to computational resource limitations. Larger models require more GPUs and significantly longer training time.

---

## License

This project is licensed under the terms specified in the **LICENSE** file. Check the file for complete details.

---

## References

[1] A. Vaswani et al., "Attention is all you need," in *Proc. 31st Conf. Neural Inf. Process. Syst. (NIPS)*, Long Beach, CA, USA, 2017, pp. 5998-6008.

[2] T. B. Brown et al., "Language models are few-shot learners," in *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 1877-1901.

[3] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language models are unsupervised multitask learners," *OpenAI Blog*, vol. 1, no. 8, p. 9, 2019.

[4] D. Hendrycks and K. Gimpel, "Gaussian error linear units (GELUs)," *arXiv preprint arXiv:1606.08415*, 2016.

[5] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 1, pp. 1929-1958, 2014.

[6] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," *arXiv preprint arXiv:1607.06450*, 2016.

**Course:** ECE 518: Neural Networks

**Training Data:** *Outliers* by Malcolm Gladwell (utilized solely for experimentation).