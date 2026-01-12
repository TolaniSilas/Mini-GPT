import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path


def train_epoch(model, dataloader, optimizer, criterion, device):
    """trains model for one epoch."""

    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):

        # move data to device (device - gpu or cpu).
        inputs = inputs.to(device)
        targets = targets.to(device)

        # forward pass.
        optimizer.zero_grad()
        outputs = model(inputs)

        # calculate loss.
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        # backward pass.
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # print progress.
        if (batch_idx + 1) % 100 == 0:
            print(f"batch {batch_idx + 1}/{len(dataloader)}, loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches

    return avg_loss


def validate(model, dataloader, criterion, device):
    """validates model on validation set."""

    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:

            # move data to device (device - gpu or cpu).
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward pass.
            outputs = model(inputs)

            # calculate loss.
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    return avg_loss


def main():
    """main training loop."""

    parser = argparse.ArgumentParser(description="train gpt-2 model")

    parser.add_argument("--data_path", type=str, default="data/processed", help="path to processed data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")

    args = parser.parse_args()

    # create checkpoint directory.
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # set device.
    device = torch.device(args.device)
    print(f"using device: {device}\n")


    from src.tokenizers import BPETokenizer
    from src.utils.config import GPT2_SMALL_124M
    from src.models.gpt import GPTModel
    from src.data.dataset import GPTTextDataset



    # initialize tokenizer.
    tokenizer = BPETokenizer()

    # load gpt-2 small configuration.
    config = GPT2_SMALL_124M

    # initialize model.
    # model = GPTModel(config).to(device)
    # print(f"model parameters: {count_parameters(model):,}\n")


    # create dataloaders.
    train_dataset = GPTTextDataset(args.data_path, tokenizer)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # initialize optimizer and loss.
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.CrossEntropyLoss()

    # training loop.
    print("starting training...\n")

    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")

        # train for one epoch.
        # train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        # print(f"train loss: {train_loss:.4f}")

        # save checkpoint.
        checkpoint_path = f"{args.checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pt"
        # from src.utils.helpers import save_checkpoint
        # save_checkpoint(model, optimizer, epoch + 1, train_loss, checkpoint_path)

        print()

    print("training complete!")


if __name__ == "__main__":
    main()