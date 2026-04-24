import os
import PyPDF2
import pandas as pd

pdf_dir = 'artikel'
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
data = []

for idx, pdf in enumerate(pdf_files):
    text_content = ""
    try:
        pdf_path = os.path.join(pdf_dir, pdf)
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + " "
    except Exception as e:
        print(f"Failed to read {pdf}: {e}")
        text_content = "Blank or unreadable PDF."
    
    # Process text: simplify spaces
    text_content = " ".join(text_content.split())
    # Format entry
    formatted_text = f"[{pdf}] {text_content}"
    data.append([idx + 1, formatted_text])

df = pd.DataFrame(data, columns=["ID", "Isi Dokumen"])
df.to_csv('dataset_berita_teknologi.csv', index=False)
print("Berhasil mengekstrak teks dari 10 PDF ke dataset_berita_teknologi.csv!")
