import os
import json
from pdfminer.high_level import extract_text

pdf_dir = "./PDFs"
output = []

for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_dir, filename)
        print(f"📄 Reading {filename}")
        try:
            text = extract_text(path)
            output.append({
                "filename": filename,
                "text": text.strip()
            })
        except Exception as e:
            print(f"⚠️ Failed to process {filename}: {e}")

# Write all to a single JSON file
with open("pdf_texts.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("✅ Done! Saved as pdf_texts.json")

