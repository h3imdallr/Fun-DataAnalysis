"""
Author: h3idallar
ref = http://wwwhihaho.synology.me/hoon/?p=418
"""

import numpy as np
import pandas as pd
import requests
import urllib.request
from bs4 import BeautifulSoup

ADDRESS = 11110
YEARMONTH = 201702
KEY = "E5ejsJWFdqc8CjN4UU0EN49vjjTz5NiM%2FtWIMasPNMmOs2wd%2F1zibZznJR9My2Gi2BscReMX44sbPxzxH57iYA%3D%3D"
URL = "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade?LAWD_CD=" \
      + str(ADDRESS) + "&DEAL_YMD=" + str(YEARMONTH) + "&serviceKey=" + KEY
COL = ['거래금액', '건축년도', '년', '법정동', '아파트', '월', '일', '전용면적', '지번', '지역코드', '층', 'Null_col']
RE_COL = ['년','월', '일', '건축년도', '아파트', '법정동', '거래금액', '전용면적', '지역코드', '지번', '층']
CLEAR_COL = ['거래금액', '건축년도',  '법정동', '아파트', '전용면적', '지번', '지역코드','년', '월', '일',  '층']


def URL_update(addr,yearmonth):
    URL = "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade?LAWD_CD=" \
          + str(addr) + "&DEAL_YMD=" + str(yearmonth) + "&serviceKey=" + KEY
    return URL


def getaptdata(url):
    f = urllib.request.urlopen(url)
    get_apt = f.read().decode("utf8")
    f.close()

    soup = BeautifulSoup(get_apt, "lxml")
    apt = list(apt.get_text().replace('\n', '').split(">") for apt in soup.find_all("item"))

    # customized by h3imdallr
    apt_df = pd.DataFrame.from_records(apt, columns= COL)
    apt_df.drop('Null_col', axis=1, inplace=True)

    for word in CLEAR_COL:
        apt_df = apt_df.applymap(lambda x: str.replace(x,word,""))

    apt_df = apt_df[RE_COL]

    return (apt_df)


# months = [1,2,3,4,5,6,7,8,9,10,11,12]
months = [1,2,3,4,5]
apt_data_0 = pd.DataFrame(columns = RE_COL) # empty DF
for yr in range(2015,2017):
    for mon in range(1,13):
        YEARMONTH =yr*100+mon
        URL = URL_update(ADDRESS, YEARMONTH)

        apt_data = getaptdata(URL)
        apt_data_0 = pd.concat([apt_data_0, apt_data])


print (apt_data_0)

# save into csv below..

