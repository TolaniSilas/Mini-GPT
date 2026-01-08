# Mini-GPT
This repository implements the **Generative Pretrained Transformer (GPT)** series (e.g., GPT-2, GPT-3, and later variants). The model architecture is the foundational building block underlying the GPT family, and its design is inspired by a major breakthrough in deep learning: the **self-attention mechanism** introduced in “Attention Is All You Need” by Vaswani et al. 

The architecture utilizes a **decoder-only Transformer** design and incorporates robust text-preprocessing techniques, such as **Byte Pair Encoding (BPE)** tokenization. BPE enables the model to effectively handle words that were not present in the training dataset by decomposing them into subword units, thereby improving vocabulary coverage and generalization.

**TL;DR:** This repository includes the coding and implementation from scratch of the following papers:[*Language Models are Unsupervi sed Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [*Language Models are Few-Shot Learners*](https://arxiv.org/pdf/2005.14165), [*Attention Is All You Need*](https://arxiv.org/pdf/1706.03762)
