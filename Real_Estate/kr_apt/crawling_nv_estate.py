"""https://financedata.github.io/posts/naver-land-crawling.html"""




import re
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from bs4 import BeautifulSoup


def get_naver_realasset(area_code, page=1):
    url = 'http://land.naver.com/article/articleList.nhn?' \
          + 'rletTypeCd=A01&tradeTypeCd=A1&hscpTypeCd=A01%3AA03%3AA04' \
          + '&cortarNo=' + area_code \
          + '&page=' + str(page)

    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')

    table = soup.find('table')
    trs = table.tbody.find_all('tr')
    if '등록된 매물이 없습니다' in trs[0].text:
        return pd.DataFrame()

    value_list = []

    # 거래, 종류, 확인일자, 매물명, 면적(㎡), 층, 매물가(만원), 연락처
    for tr in trs[::2]:
        tds = tr.find_all('td')
        cols = [' '.join(td.text.strip().split()) for td in tds]

        if '_thumb_image' not in tds[3]['class']:  # 현장확인 날짜와 이미지가 없는 행
            cols.insert(3, '')

        # print(cols)
        거래 = cols[0]
        종류 = cols[1]
        확인일자 = datetime.strptime(cols[2], '%y.%m.%d.')
        현장확인 = cols[3]
        매물명 = cols[4]
        면적 = cols[5]
        공급면적 = re.findall('공급면적(.*?)㎡', 면적)[0].replace(',', '')
        전용면적 = re.findall('전용면적(.*?)㎡', 면적)[0].replace(',', '')
        공급면적 = float(공급면적)
        전용면적 = float(전용면적)
        층 = cols[6]
        if cols[7].find('호가일뿐 실거래가로확인된 금액이 아닙니다') >= 0:
            pass  # 단순호가 별도 처리하고자 하면 내용 추가
        매물가 = int(cols[7].split(' ')[0].replace(',', ''))
        연락처 = cols[8]

        value_list.append([거래, 종류, 확인일자, 현장확인, 매물명, 공급면적, 전용면적, 층, 매물가, 연락처])

    cols = ['거래', '종류', '확인일자', '현장확인', '매물명', '공급면적', '전용면적', '층', '매물가', '연락처']
    df = pd.DataFrame(value_list, columns=cols)
    return df


# df = get_naver_realasset('1168010600', 10) # 10 페이지
# print(df.tail())

area_code = '1168010600' # 강남구, 대치동 (법정동 코드 https://goo.gl/P6ni8Q 참조)

df = pd.DataFrame()
for i in range(1, 10): # 최대 100 페이지
    df_tmp = get_naver_realasset(area_code, i)
    if len(df_tmp) <= 0:
        break
    df = df.append(df_tmp, ignore_index=True)

df.head()