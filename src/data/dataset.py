import os
from pathlib import Path
from src.tokenizers import BPETokenizer
import torch
from torch.utils.data import Dataset, DataLoader


class GPTTextDataset(Dataset):
    """pytorch custom dataset for tokenized text data."""

    def __init__(self, data_path, tokenizer, context_length=256, max_length=200, stride=128):
        """initializes dataset with text files and tokenizer."""

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.max_length = max_length
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

            # tokenize the entire text (document).
            token_ids = self.tokenizer.encode(text)

            # create a sliding window to chunk the text (document) into overlapping sequences of context length.
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

        # convert to tensors with long data type.
        input_tensor = torch.tensor(self.input_ids[idx], dtype=torch.long)
        target_tensor = torch.tensor(self.target_ids[idx], dtype=torch.long)

        return input_tensor, target_tensor



def create_dataloader(dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=0):
    """creates pytorch dataloader from the custom dataset."""

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader



if __name__ == "__main__":
  
    # initialize tokenizer.
    tokenizer = BPETokenizer("gpt2")                     

    # create dataset.
    dataset = GPTTextDataset(
        data_path="data/processed",
        tokenizer=tokenizer,
        context_length=256,
        stride=128
    )

    # create dataloader.
    dataloader = create_dataloader(dataset, batch_size=8, shuffle=True)

    # running iteration for text.
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"batch {batch_idx + 1}:")
        print(f"  input shape: {inputs.shape}")
        print(f"  target shape: {targets.shape}")

        if batch_idx >= 2:
            break