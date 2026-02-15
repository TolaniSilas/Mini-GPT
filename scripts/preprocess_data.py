import sys
from pathlib import Path

# add project root to python path.
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse
from src.data import pdf_to_text, batch_pdf_to_text



def main():
    """main preprocessing pipeline."""

    parser = argparse.ArgumentParser(description="preprocess pdf data for gpt-2 training.")

    parser.add_argument("--raw_dir", type=str, default="data/raw", help="directory with raw pdfs")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="output directory for processed text")

    args = parser.parse_args()

    # create output directories.
    os.makedirs(args.output_dir, exist_ok=True)

    # get all pdf files.
    pdf_files = list(Path(args.raw_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"no pdf files found in {args.raw_dir}")
        return
    
    if len(pdf_files) == 1:
        print("found 1 pdf file\n.")
    
    else:
        print(f"found {len(pdf_files)} pdf files\n")


    # process each pdf.
    batch_pdf_to_text(args.raw_dir, args.output_dir)

    print("\npreprocessing complete!")


if __name__ == "__main__":

    # run the main preprocessing pipeline.
    main()