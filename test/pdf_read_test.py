from custom_util import read_pdf


clean_text = read_pdf.read_pdf_to_clean_text('safely.pdf')
print(clean_text)

clean_text_2 = read_pdf.read_pdf_to_clean_text('AI-tech.pdf')
print(clean_text_2)

