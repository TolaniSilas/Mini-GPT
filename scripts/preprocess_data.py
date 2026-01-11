import os
import re
import argparse
from pathlib import Path


def preprocess_text(text):
    """cleans and preprocesses raw text."""

    # remove extra whitespace.
    text = re.sub(r'\s+', ' ', text)

    # remove leading/trailing whitespace.
    text = text.strip()

    return text


def process_pdf_to_text(pdf_path, output_path):
    """converts pdf to preprocessed text file."""

    try:
        # import pdf reader (assuming it exists in src.data).
        from src.data.pdf_reader import pdf_to_text

        # extract text from pdf.
        print(f"processing: {pdf_path}")
        raw_text = pdf_to_text(pdf_path, save_to_file=False)

        if raw_text:
            # preprocess text.
            cleaned_text = preprocess_text(raw_text)

            # save to output file.
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            print(f"saved to: {output_path}")
            return True
        else:
            print(f"failed to extract text from: {pdf_path}")
            return False

    except Exception as e:
        print(f"error processing {pdf_path}: {e}")
        return False


def build_vocabulary(text_files, min_freq=2, special_tokens=None):
    """builds vocabulary from text files."""

    if special_tokens is None:
        special_tokens = ["<|unk|>", "<|endoftext|>"]

    # count word frequencies.
    word_freq = {}

    for text_file in text_files:
        print(f"reading: {text_file}")

        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # tokenize (simple whitespace split for vocab building).
        words = text.split()

        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

    # filter by minimum frequency.
    vocab = {word: idx for idx, word in enumerate(special_tokens)}
    idx = len(special_tokens)

    for word, freq in sorted(word_freq.items()):
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    print(f"\nvocabulary size: {len(vocab)}")

    return vocab


def main():
    """main preprocessing pipeline."""

    parser = argparse.ArgumentParser(description="preprocess data for gpt-2 training")

    parser.add_argument("--raw_dir", type=str, default="data/raw", help="directory with raw pdfs")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="output directory for processed text")
    parser.add_argument("--vocab_path", type=str, default="data/vocab/vocab.json", help="path to save vocabulary")
    parser.add_argument("--min_freq", type=int, default=2, help="minimum word frequency for vocab")

    args = parser.parse_args()

    # create output directories.
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.vocab_path), exist_ok=True)

    # get all pdf files.
    pdf_files = list(Path(args.raw_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"no pdf files found in {args.raw_dir}")
        return

    print(f"found {len(pdf_files)} pdf files\n")

    # process each pdf.
    processed_files = []

    for pdf_file in pdf_files:
        output_file = Path(args.output_dir) / f"{pdf_file.stem}.txt"

        if process_pdf_to_text(str(pdf_file), str(output_file)):
            processed_files.append(str(output_file))

    # build vocabulary from processed files.
    if processed_files:
        print("\nbuilding vocabulary...")
        vocab = build_vocabulary(processed_files, min_freq=args.min_freq)

        # save vocabulary.
        from src.utils.helpers import save_vocab
        save_vocab(vocab, args.vocab_path)

    print("\npreprocessing complete!")


if __name__ == "__main__":
    main()