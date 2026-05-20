"""
24. ファイル参照の抽出 / 提取文件引用
記事から参照されているメディアファイルをすべて抜き出す。
从文章中提取所有被引用的媒体文件。

MediaWiki のファイル参照 / MediaWiki 的文件引用：
  [[ファイル:XXX.jpg|...]]
  [[File:XXX.png|...]]
  [[Image:XXX.svg|...]]
"""

import re
from knock20 import load_uk_article


def extract_media_files(text: str) -> list[str]:
    """
    メディアファイル名のリストを返す。
    返回媒体文件名的列表。
    """
    # ファイル / File / Image のいずれかの接頭辞にマッチ
    # 匹配以「ファイル」/「File」/「Image」开头的引用
    pattern = re.compile(
        r"\[\[(?:ファイル|File|Image):([^|\]]+?)(?:\|.*?)?\]\]"
    )
    return pattern.findall(text)

'''
\[\[                        # 字面量 [[
(?:ファイル|File|Image)    # 非捕获组：ファイル または File または Image
:                           # 字面量 :
([^|\]]+?)                  # キャプチャグループ：| と ] 以外の文字（非貪欲）
(?:\|.*?)?                  # 非捕获組：| で始まるパラメータ（0回または1回）
\]\]                        # 字面量 ]]
'''
if __name__ == "__main__":
    # 本文取得 / 取得正文
    text = load_uk_article()

    # ファイル参照を抽出 / 提取文件引用
    files = extract_media_files(text)

    # 表示 / 输出
    for f in files:
        print(f)

'''
Royal Coat of Arms of the United Kingdom.svg
United States Navy Band - God Save the Queen.ogg
Descriptio Prime Tabulae Europae.jpg
Lenepveu, Jeanne d'Arc au siège d'Orléans.jpg
London.bankofengland.arp.jpg
Battle of Waterloo 1815.PNG
Uk topo en.jpg
BenNevis2005.jpg
Population density UK 2011 census.png
2019 Greenwich Peninsula & Canary Wharf.jpg
Birmingham Skyline from Edgbaston Cricket Ground crop.jpg
Leeds CBD at night.jpg
Glasgow and the Clyde from the air (geograph 4665720).jpg
Palace of Westminster, London - Feb 2007.jpg
Scotland Parliament Holyrood.jpg
Donald Trump and Theresa May (33998675310) (cropped).jpg
Soldiers Trooping the Colour, 16th June 2007.jpg
City of London skyline from London City Hall - Oct 2008.jpg
Oil platform in the North SeaPros.jpg
Eurostar at St Pancras Jan 2008.jpg
Heathrow Terminal 5C Iwelumo-1.jpg
Airbus A380-841 G-XLEB British Airways (10424102995).jpg
UKpop.svg
Anglospeak.svg
Royal Aberdeen Children's Hospital.jpg
CHANDOS3.jpg
The Fabs.JPG
Wembley Stadium, illuminated.jpg
'''