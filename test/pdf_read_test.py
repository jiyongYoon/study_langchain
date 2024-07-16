# from custom_util import read_pdf
#
#
# clean_text = read_pdf.read_pdf_to_clean_text('safely.pdf')
# print(clean_text)
#
# clean_text_2 = read_pdf.read_pdf_to_clean_text('AI-tech.pdf')
# print(clean_text_2)
#

from load_data import pdf_loader

pdf_name = 'page-test.pdf'

pdf = pdf_loader.load_pdf(pdf_name) ## Document 객체

for page in pdf:
    print(page)
    """
    page_content='01총론 1\n02정의(중대산업재해 , 경영책임자 , 종사자)  7\n03법 적용 범위 및 시기 1 5\n04안전 및 보건 확보의무  2 3\n전 담  조 직, 유 해･위 험 요 인  확 인  및  개 선  점 검, 예 산  편 성  및  집 행, 종 사 자  의 견  청 취  등\n05경영책임자등에 대한 처벌  4 7' metadata={'source': 'D:\\\\dev_yoon\\\\py\\\\study_langchain\\page-test.pdf', 'page': 2}
    page_content='총론 ⚫⚫⚫1\n총론1' metadata={'source': 'D:\\\\dev_yoon\\\\py\\\\study_langchain\\page-test.pdf', 'page': 4}
    """