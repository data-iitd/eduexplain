import fitz
import os

PDF_ROOT = "./NCERT"

OUTPUT_ROOT = "./page_images"

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)

for root, dirs, files in os.walk(PDF_ROOT):

    for file in files:

        if not file.endswith(".pdf"):
            continue

        pdf_path = os.path.join(
            root,
            file
        )

        print("Processing:", pdf_path)

        pdf = fitz.open(pdf_path)

        book_name = file.replace(
            ".pdf",
            ""
        )

        book_folder = os.path.join(
            OUTPUT_ROOT,
            book_name
        )

        os.makedirs(
            book_folder,
            exist_ok=True
        )

        for page_num in range(len(pdf)):

            page = pdf[page_num]

            pix = page.get_pixmap(
                matrix=fitz.Matrix(3,3),
                alpha=False
            )

            pix.save(
                os.path.join(
                    book_folder,
                    f"page_{page_num+1}.png"
                )
            )

        pdf.close()

print("Done")