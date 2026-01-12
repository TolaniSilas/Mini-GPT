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


    # decode generated ids.
    generated_text = tokenizer.decode(generated_ids)

    return generated_text


def main():
    """main generation script."""

    parser = argparse.ArgumentParser(description="generate text with custom akin gpt-2 model trained on outliers' book")

    parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="who is the author of the outliers book?", help="prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100, help="maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")

    args = parser.parse_args()

    # set device.
    device = torch.device(args.device)
    print(f"using device: {device}\n")

    
    from src.tokenizers import BPETokenizer
    from src.utils.config import GPT2_SMALL_124M
    from src.models.gpt import GPTModel

    # initialize tokenizer.
    tokenizer = BPETokenizer()

    # load gpt-2 small configuration.
    config = GPT2_SMALL_124M

    # initialize model.
    # model = GPTModel(config).to(device)

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