"""
25. テンプレートの抽出 / 提取模板
記事中の「基礎情報」テンプレートのフィールド名と値を抽出し、
辞書オブジェクトとして格納する。
提取文章中"基础信息（基礎情報）"模板的字段名和值，存储为字典对象。

テンプレートの形式 / 模板的格式：
  {{基礎情報 国
  |略名 = イギリス
  |日本語国名 = グレートブリテン...
  ...
  }}
"""

import re
from knock20 import load_uk_article


def extract_basic_info(text: str) -> dict[str, str]:
    """
    「基礎情報」テンプレートを辞書として返す。
    将"基础信息"模板作为字典返回。
    """
    # テンプレート本体を取り出す / 提取模板主体
    # {{基礎情報 ... }} の中身を非貪欲にマッチ
    # 非贪婪地匹配 {{基礎情報 ... }} 的内容
    template_match = re.search(
        r"\{\{基礎情報.*?\n(.*?)\n\}\}",
        text,
        re.DOTALL,
    )
    if not template_match:
        return {}

    body = template_match.group(1)

    # フィールドを分割 / 分割字段
    # 行頭 "|キー = 値" にマッチ、次の "|キー =" まで（または末尾まで）
    # 匹配行首的 "|键 = 值"，直到下一个 "|键 =" 或文本末尾
    field_pattern = re.compile(
        r"^\|([^=\s]+)\s*=\s*(.*?)(?=\n\|[^=\s]+\s*=|\Z)",
        re.MULTILINE | re.DOTALL,
    )
# ^                              # 行の始まり
# \|                             # 字面量 |
# ([^=\s]+)                      # キャプチャグループ 1: = とスペース以外の文字（パラメータ名）
# \s*                            # スペース 0回以上
# =                              # 字面量 =
# \s*                            # スペース 0回以上
# (.*?)                          # キャプチャグループ 2: 任意の文字（非貪欲、パラメータ値）
# (?=                            # 正向先読み（これ以後の文字を消費しない）
#   \n\|[^=\s]+\s*=             # 次の行の新しいパラメータ
#   |                            # または
#   \Z                           # 文字列末尾
# )
    info: dict[str, str] = {}  # 空の辞書を作成
    for m in field_pattern.finditer(body):  # 各フィールドをマッチ
        key = m.group(1).strip()  # パラメータ名を取得して余白を削除
        value = m.group(2).strip()  # パラメータ値を取得して余白を削除
        info[key] = value  # 辞書に追加
    return info  # 辞書を返す


if __name__ == "__main__":
    # 本文取得 / 取得正文
    text = load_uk_article()

    # 基礎情報を抽出 / 提取基础信息
    info = extract_basic_info(text)

    # 表示 / 输出
    for k, v in info.items():
        print(f"{k}: {v}")

'''
略名: イギリス
日本語国名: グレートブリテン及び北アイルランド連合王国
公式国名: {{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />
*{{lang|gd|An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath}}（[[スコットランド・ゲール語]]）
*{{lang|cy|Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon}}（[[ウェールズ語]]）
*{{lang|ga|Ríocht Aontaithe na Breataine Móire agus Tuaisceart na hÉireann}}（[[アイルランド語]]）
*{{lang|kw|An Rywvaneth Unys a Vreten Veur hag Iwerdhon Glédh}}（[[コーンウォール語]]）
*{{lang|sco|Unitit Kinrick o Great Breetain an Northren Ireland}}（[[スコットランド語]]）
**{{lang|sco|Claught Kängrick o Docht Brätain an Norlin Airlann}}、{{lang|sco|Unitet Kängdom o Great Brittain an Norlin Airlann}}（アルスター・スコットランド語）</ref>
国旗画像: Flag of the United Kingdom.svg
国章画像: [[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]
国章リンク: （[[イギリスの国章|国章]]）
標語: {{lang|fr|[[Dieu et mon droit]]}}<br />（[[フランス語]]:[[Dieu et mon droit|神と我が権利]]）
国歌: [[女王陛下万歳|{{lang|en|God Save the Queen}}]]{{en icon}}<br />''神よ女王を護り賜え''<br />{{center|[[ファイル:United States Navy Band - God Save the Queen.ogg]]}}
地図画像: Europe-UK.svg
位置画像: United Kingdom (+overseas territories) in the World (+Antarctica claims).svg
公用語: [[英語]]
首都: [[ロンドン]]（事実上）
最大都市: ロンドン
元首等肩書: [[イギリスの君主|女王]]
元首等氏名: [[エリザベス2世]]
首相等肩書: [[イギリスの首相|首相]]
首相等氏名: [[ボリス・ジョンソン]]
他元首等肩書1: [[貴族院 (イギリス)|貴族院議長]]
他元首等氏名1: [[:en:Norman Fowler, Baron Fowler|ノーマン・ファウラー]]
他元首等肩書2: [[庶民院 (イギリス)|庶民院議長]]
他元首等氏名2: {{仮リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}
他元首等肩書3: [[連合王国最高裁判所|最高裁判所長官]]
他元首等氏名3: [[:en:Brenda Hale, Baroness Hale of Richmond|ブレンダ・ヘイル]]
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
確立形態1: [[イングランド王国]]／[[スコットランド王国]]<br />（両国とも[[合同法 (1707年)|1707年合同法]]まで）
確立年月日1: 927年／843年
確立形態2: [[グレートブリテン王国]]成立<br />（1707年合同法）
確立年月日2: 1707年{{0}}5月{{0}}1日
確立形態3: [[グレートブリテン及びアイルランド連合王国]]成立<br />（[[合同法 (1800年)|1800年合同法]]）
確立年月日3: 1801年{{0}}1月{{0}}1日
確立形態4: 現在の国号「''グレートブリテン及び北アイルランド連合王国''」に変更
確立年月日4: 1927年{{0}}4月12日
通貨: [[スターリング・ポンド|UKポンド]] (£)
通貨コード: GBP
時間帯: ±0
夏時間: +1
|ISO 3166-1 = GB / GBR
ccTLD: [[.uk]] / [[.gb]]<ref>使用は.ukに比べ圧倒的少数。</ref>
国際電話番号: 44
注記: <references/>
'''