from pypdf import PdfReader # 推荐使用新的 pypdf 库，它是 PyPDF2 的继任者

def extract_text_with_pypdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

text = extract_text_with_pypdf("./downloads/0b9d0bee85e4ef4261147f35be885010e62ad1fb.pdf")
print(len(text))