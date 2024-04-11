import pandas as pd

# XLSX 파일을 읽기
df = pd.read_excel('sydney_tem.xlsx')

# CSV 파일로 저장
df.to_csv('sydney_tem.csv', index=False)
