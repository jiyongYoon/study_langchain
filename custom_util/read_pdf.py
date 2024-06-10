### PDF
import os
import pdfplumber
import re


def remove_special_characters(text):
    # 특수 문자 제거 (정규 표현식 사용)
    return re.sub(r'[^A-Za-z0-9가-힣\s]', '', text)


def read_pdf_to_clean_text(pdf_filename):
    # project_directory = 'D:\\dev_yoon\\py\\study_langchain'
    project_directory = os.environ['PROJECT_DIRECTORY']
    pdf_filepath = os.path.join(project_directory, pdf_filename)

    pdf_content = ""
    try:
        with pdfplumber.open(pdf_filepath) as pdf:
            for page in pdf.pages:
                pdf_content += page.extract_text() + "\n"
        print("PDF 파일을 성공적으로 로드하고 텍스트를 추출했습니다.")
    except FileNotFoundError:
        print(f"PDF 파일 '{pdf_filename}'을 찾을 수 없습니다.")
    except Exception as e:
        print(f"PDF 파일을 읽는 중 오류 발생: {e}")

    cleaned_text = remove_special_characters(pdf_content)

    return cleaned_text