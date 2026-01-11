# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path


class TextDataset(Dataset):
    """pytorch dataset for tokenized text data."""

    def __init__(self, data_path, tokenizer, context_length=256, stride=128):
        """initializes dataset with text files and tokenizer."""

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride

        # load all text files.
        self.text_files = self._load_text_files(data_path)

        # tokenize all text and create chunks.
        self.input_ids = []
        self.target_ids = []

        self._prepare_data()

    def _load_text_files(self, data_path):
        """loads all text files from directory."""

        text_files = []

        if os.path.isfile(data_path):
            # single file.
            text_files.append(data_path)
        elif os.path.isdir(data_path):
            # directory of files.
            text_files = list(Path(data_path).glob("*.txt"))
        else:
            raise ValueError(f"invalid data path: {data_path}")

        print(f"found {len(text_files)} text files")

        return text_files

    def _prepare_data(self):
        """tokenizes text and creates input-target pairs."""

        for text_file in self.text_files:
            print(f"processing: {text_file}")

            # read text file.
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # tokenize text.
            token_ids = self.tokenizer.encode(text)

            # create sliding window chunks.
            for i in range(0, len(token_ids) - self.context_length, self.stride):
                # input is current chunk.
                input_chunk = token_ids[i:i + self.context_length]

                # target is shifted by one position.
                target_chunk = token_ids[i + 1:i + self.context_length + 1]

                # ensure both chunks are same length.
                if len(input_chunk) == self.context_length and len(target_chunk) == self.context_length:
                    self.input_ids.append(input_chunk)
                    self.target_ids.append(target_chunk)

        print(f"created {len(self.input_ids)} training samples")

    def __len__(self):
        """returns the number of samples in dataset."""

        return len(self.input_ids)

    def __getitem__(self, idx):
        """returns input and target tensors for given index."""

        # convert to tensors.
        input_tensor = torch.tensor(self.input_ids[idx], dtype=torch.long)
        target_tensor = torch.tensor(self.target_ids[idx], dtype=torch.long)

        return input_tensor, target_tensor


def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=0):
    """creates pytorch dataloader from dataset."""

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


# example usage.
if __name__ == "__main__":
    # example of how to use the dataset.
    from src.tokenizers.word_tokenizer import WordTokenizer
    from src.utils.helpers import load_vocab

    # load vocabulary.
    vocab = load_vocab("data/vocab/vocab.json")

    if vocab:
        # initialize tokenizer.
        tokenizer = WordTokenizer(vocab)

        # create dataset.
        dataset = TextDataset(
            data_path="data/processed",
            tokenizer=tokenizer,
            context_length=256,
            stride=128
        )

        # create dataloader.
        dataloader = create_dataloader(dataset, batch_size=8, shuffle=True)

        # test iteration.
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(f"batch {batch_idx + 1}:")
            print(f"  input shape: {inputs.shape}")
            print(f"  target shape: {targets.shape}")

            if batch_idx >= 2:
                break