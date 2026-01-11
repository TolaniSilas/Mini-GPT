import os
import PyPDF2


def pdf_to_text(pdf_path, output_txt_path=None, save_to_file=True):
    """reads pdf and converts to text, optionally saving to .txt file."""

    # check if pdf exists.
    if not os.path.exists(pdf_path):
        print(f"error: '{pdf_path}' not found.")
        return None

    try:
        # open pdf in binary mode.
        with open(pdf_path, 'rb') as file:

            # create pdf reader.
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"processing {num_pages} pages...")

            # extract all text.
            full_text = []

            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()

                if text:
                    full_text.append(text)
                else:
                    print(f"warning: page {page_num + 1} had no extractable text.")

            # combine all pages.
            combined_text = "\n\n".join(full_text)

            # save to file if requested.
            if save_to_file:
                if output_txt_path is None:
                    output_txt_path = pdf_path.replace('.pdf', '.txt')

                with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(combined_text)

                print(f"saved to: {output_txt_path}")

            return combined_text

    except Exception as e:
        print(f"error reading pdf: {e}")
        return None


def batch_pdf_to_text(folder_path, output_folder=None):
    """converts all pdfs in a folder to text files."""

    # check if folder exists.
    if not os.path.exists(folder_path):
        print(f"error: folder '{folder_path}' not found.")
        return

    # create output folder if needed.
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # get all pdf files.
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    if not pdf_files:
        print("no pdf files found in folder.")
        return

    print(f"found {len(pdf_files)} pdf files. converting...")

    # process each pdf.
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)

        if output_folder:
            output_path = os.path.join(output_folder, pdf_file.replace('.pdf', '.txt'))
        else:
            output_path = None

        print(f"\nprocessing: {pdf_file}")
        pdf_to_text(pdf_path, output_path)

    print("\nbatch conversion complete!")