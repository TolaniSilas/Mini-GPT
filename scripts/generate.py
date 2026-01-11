import torch
import argparse


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=1.0, device="cpu"):
    """generates text from a prompt using the trained model."""

    model.eval()

    # encode prompt.
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)

    generated_ids = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_tokens):
            # get model predictions.
            outputs = model(input_tensor)

            # get last token logits.
            logits = outputs[0, -1, :] / temperature

            # sample next token.
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # append to generated sequence.
            generated_ids.append(next_token)

            # update input tensor.
            input_tensor = torch.tensor([generated_ids]).to(device)

            # stop if end token is generated.
            if next_token == tokenizer.tok_to_int.get("<|endoftext|>"):
                break

    # decode generated ids.
    generated_text = tokenizer.decode(generated_ids)

    return generated_text


def main():
    """main generation script."""

    parser = argparse.ArgumentParser(description="generate text with custom akin gpt-2 model trained on outliers' book")

    parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--vocab_path", type=str, default="data/vocab/vocab.json", help="path to vocabulary")
    parser.add_argument("--prompt", type=str, default="who is the author of the outliers book?", help="prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100, help="maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")

    args = parser.parse_args()

    # set device.
    device = torch.device(args.device)
    print(f"using device: {device}\n")

    # load vocabulary.
    from src.utils.helpers import load_vocab
    vocab = load_vocab(args.vocab_path)

    if vocab is None:
        print("failed to load vocabulary")
        return

    # initialize tokenizer.
    from src.tokenizers import BPETokenizer
    # tokenizer = WordTokenizer(vocab)
    tokenizer = BPETokenizer()

    # load model (assuming GPT2Model exists).
    from src.utils.config import GPT2_SMALL
    # from src.models.gpt2 import GPT2Model

    config = GPT2_SMALL
    config.vocab_size = len(vocab)

    # model = GPT2Model(config).to(device)

    # load checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])

    print(f"loaded checkpoint: {args.checkpoint}")
    print(f"prompt: {args.prompt}\n")

    # generate text.
    # generated_text = generate_text(
    #     model, tokenizer, args.prompt, 
    #     max_tokens=args.max_tokens,
    #     temperature=args.temperature,
    #     device=device
    # )

    # print(f"generated text:\n{generated_text}")


if __name__ == "__main__":
    main()