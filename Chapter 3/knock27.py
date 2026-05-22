"""
27. 内部リンクの除去 / 移除内部链接
26 の処理に加えて、テンプレートの値から MediaWiki の内部リンク
マークアップを除去し、テキストに変換する。
在 26 的处理基础上，从模板的值中移除 MediaWiki 的内部链接标记并转换为纯文本。

MediaWiki の内部リンク / MediaWiki 的内部链接：
  [[記事名]]              → 「記事名」
  [[記事名|表示文字]]      → 「表示文字」
  [[記事名#節|表示文字]]   → 「表示文字」
"""

import re
from knock20 import load_uk_article
from knock25 import extract_basic_info
from knock26 import remove_emphasis


def remove_internal_links(value: str) -> str:
    """
    内部リンクマークアップを除去し、表示文字（または記事名）のみ残す。
    移除内部链接标记，仅保留显示文字（或文章名）。
    """
    # [[XXX]] または [[XXX|YYY]] にマッチ
    # 匹配 [[XXX]] 或 [[XXX|YYY]]
    # ファイル/Category などはここでは無視（基礎情報内には通常入らない）
    # 这里忽略 File/Category 等（一般不在基础信息内）
    def _replace(match: re.Match) -> str:
        inner = match.group(1)
        # パイプがあれば最後の要素（表示文字）を採用
        # 有竖线则取最后一段（显示文字）
        if "|" in inner:
            return inner.split("|")[-1]
        return inner

    return re.sub(r"\[\[(.+?)\]\]", _replace, value)

#\[\[        # リテラル [[
#(.+?)       # キャプチャグループ 1: 任意の文字（非貪欲）
#\]\]        # リテラル ]]

def clean_basic_info(text: str) -> dict[str, str]:
    """
    基礎情報を抽出し、強調と内部リンクの両方を除去する。
    提取基础信息，并同时移除强调和内部链接。
    """
    info = extract_basic_info(text)
    cleaned = {}
    for k, v in info.items():
        v = remove_emphasis(v)        # 強調除去 / 移除强调
        v = remove_internal_links(v)  # 内部リンク除去 / 移除内部链接
        cleaned[k] = v
    return cleaned


if __name__ == "__main__":
    text = load_uk_article()
    info = clean_basic_info(text)
    for k, v in info.items():
        print(f"{k}: {v}")

'''
略名: イギリス
日本語国名: グレートブリテン及び北アイルランド連合王国
公式国名: {{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />
*{{lang|gd|An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath}}（スコットランド・ゲール語）
*{{lang|cy|Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon}}（ウェールズ語）
*{{lang|ga|Ríocht Aontaithe na Breataine Móire agus Tuaisceart na hÉireann}}（アイルランド語）
*{{lang|kw|An Rywvaneth Unys a Vreten Veur hag Iwerdhon Glédh}}（コーンウォール語）
*{{lang|sco|Unitit Kinrick o Great Breetain an Northren Ireland}}（スコットランド語）
**{{lang|sco|Claught Kängrick o Docht Brätain an Norlin Airlann}}、{{lang|sco|Unitet Kängdom o Great Brittain an Norlin Airlann}}（アルスター・スコットランド語）</ref>
国旗画像: Flag of the United Kingdom.svg
国章画像: イギリスの国章
国章リンク: （国章）
標語: {{lang|fr|Dieu et mon droit}}<br />（フランス語:神と我が権利）
国歌: God Save the Queen}}{{en icon}}<br />神よ女王を護り賜え<br />{{center|ファイル:United States Navy Band - God Save the Queen.ogg}}
地図画像: Europe-UK.svg
位置画像: United Kingdom (+overseas territories) in the World (+Antarctica claims).svg
公用語: 英語
首都: ロンドン（事実上）
最大都市: ロンドン
元首等肩書: 女王
元首等氏名: エリザベス2世
首相等肩書: 首相
首相等氏名: ボリス・ジョンソン
他元首等肩書1: 貴族院議長
他元首等氏名1: ノーマン・ファウラー
他元首等肩書2: 庶民院議長
他元首等氏名2: {{仮リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}
他元首等肩書3: 最高裁判所長官
他元首等氏名3: ブレンダ・ヘイル
面積順位: 76
面積大きさ: 1 E11
面積値: 244,820
水面積率: 1.3%
人口統計年: 2018
人口順位: 22
人口大きさ: 1 E7
人口値: 6643万5600<ref>{{Cite web|url=https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates|title=Population estimates - Office for National Statistics|accessdate=2019-06-26|date=2019-06-26}}</ref>
人口密度値: 271
GDP統計年元: 2012
GDP値元: 1兆5478億<ref name="imf-statistics-gdp">[http://www.imf.org/external/pubs/ft/weo/2012/02/weodata/weorept.aspx?pr.x=70&pr.y=13&sy=2010&ey=2012&scsm=1&ssd=1&sort=country&ds=.&br=1&c=112&s=NGDP%2CNGDPD%2CPPPGDP%2CPPPPC&grp=0&a=IMF>Data and Statistics>World Economic Outlook Databases>By Countrise>United Kingdom]</ref>
GDP統計年MER: 2012
GDP順位MER: 6
GDP値MER: 2兆4337億<ref name="imf-statistics-gdp" />
GDP統計年: 2012
GDP順位: 6
GDP値: 2兆3162億<ref name="imf-statistics-gdp" />
GDP/人: 36,727<ref name="imf-statistics-gdp" />
建国形態: 建国
確立形態1: イングランド王国／スコットランド王国<br />（両国とも1707年合同法まで）
確立年月日1: 927年／843年
確立形態2: グレートブリテン王国成立<br />（1707年合同法）
確立年月日2: 1707年{{0}}5月{{0}}1日
確立形態3: グレートブリテン及びアイルランド連合王国成立<br />（1800年合同法）
確立年月日3: 1801年{{0}}1月{{0}}1日
確立形態4: 現在の国号「グレートブリテン及び北アイルランド連合王国」に変更
確立年月日4: 1927年{{0}}4月12日
通貨: UKポンド (£)
通貨コード: GBP
時間帯: ±0
夏時間: +1
|ISO 3166-1 = GB / GBR
ccTLD: .uk / .gb<ref>使用は.ukに比べ圧倒的少数。</ref>
国際電話番号: 44
注記: <references/>
'''