"""
28. MediaWikiマークアップの除去 / 移除 MediaWiki 标记
27 の処理に加えて、テンプレートの値から MediaWiki マークアップを
可能な限り除去し、国の基本情報を整形する。
在 27 的处理基础上，从模板的值中尽可能移除 MediaWiki 标记，
对国家的基本信息进行整形。

除去対象 / 移除对象：
  - 外部リンク [http://...] / 外部链接
  - ファイル参照 [[ファイル:...]] / 文件引用
  - HTMLタグ <br />, <ref>...</ref> 等 / HTML 标签
  - テンプレート {{lang|en|...}} 等 / 内嵌模板
"""

import re
from knock20 import load_uk_article
from knock25 import extract_basic_info
from knock26 import remove_emphasis
from knock27 import remove_internal_links


def remove_external_links(value: str) -> str:
    """
    外部リンクを除去する。[URL 表示文字] → 表示文字 / [URL] → 空
    移除外部链接。
    """
    # [URL 表示文字] 形式 / [URL 显示文字] 形式
    value = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", value)
    # [URL] のみの形式 / 仅 [URL] 形式
    value = re.sub(r"\[https?://[^\]]+\]", "", value)
    return value


def remove_file_refs(value: str) -> str:
    """
    [[ファイル:...]] / [[File:...]] / [[Image:...]] を除去する。
    移除文件引用。
    """
    return re.sub(r"\[\[(?:ファイル|File|Image):[^\]]+\]\]", "", value)


def remove_html_tags(value: str) -> str:
    """
    HTMLタグ（<ref>...</ref>, <br />, <ref name="x" /> など）を除去する。
    移除 HTML 标签（包括 <ref>...</ref>、<br />、<ref name="x" /> 等）。
    """
    # <ref ...>...</ref> ペアタグ / 成对的 ref 标签
    value = re.sub(r"<ref[^>]*>.*?</ref>", "", value, flags=re.DOTALL)
    # 自己閉じタグや単独タグ / 自闭合或单独标签
    value = re.sub(r"<[^>]+/?>", "", value)
    return value


def remove_templates(value: str) -> str:
    """
    {{...}} 形式のインラインテンプレートを最後の要素に置き換える。
    例：{{lang|en|United Kingdom}} → United Kingdom
    将 {{...}} 形式的内嵌模板替换为最后一段。
    """
    # 内部にネストが無いと仮定 / 假定内部无嵌套
    def _replace(match: re.Match) -> str:
        inner = match.group(1)
        parts = inner.split("|")
        # 単純なテンプレートは消す、複合は最終要素を残す
        # 简单模板移除，复合模板保留最后一段
        return parts[-1] if len(parts) > 1 else ""

    # 内部に {{ }} を含まない最も内側のテンプレートを繰り返し処理
    # 反复处理最内层、不含 {{ }} 的模板
    prev = None
    while prev != value:
        prev = value
        value = re.sub(r"\{\{([^{}]+?)\}\}", _replace, value)
    return value


def clean_value(value: str) -> str:
    """
    各種マークアップを順番に除去する。
    依次移除各种标记。
    """
    value = remove_emphasis(value)         # '''強調''' / 强调
    value = remove_file_refs(value)        # [[ファイル:...]] / 文件
    value = remove_internal_links(value)   # [[内部リンク]] / 内部链接
    value = remove_external_links(value)   # [外部リンク] / 外部链接
    value = remove_html_tags(value)        # <ref>等 / HTML 标签
    value = remove_templates(value)        # {{lang|en|...}} / 内嵌模板
    return value.strip()


def clean_basic_info(text: str) -> dict[str, str]:
    """
    基礎情報を抽出し、すべてのマークアップを除去する。
    提取基础信息并移除所有标记。
    """
    info = extract_basic_info(text)
    return {k: clean_value(v) for k, v in info.items()}


if __name__ == "__main__":
    text = load_uk_article()
    info = clean_basic_info(text)
    for k, v in info.items():
        print(f"{k}: {v}")

'''
略名: イギリス
日本語国名: グレートブリテン及び北アイルランド連合王国
公式国名: United Kingdom of Great Britain and Northern Ireland
国旗画像: Flag of the United Kingdom.svg
国章画像: 
国章リンク: （国章）
標語: Dieu et mon droit（フランス語:神と我が権利）
国歌: God Save the Queen}}神よ女王を護り賜え
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
他元首等氏名2: Lindsay Hoyle
他元首等肩書3: 最高裁判所長官
他元首等氏名3: ブレンダ・ヘイル
面積順位: 76
面積大きさ: 1 E11
面積値: 244,820
水面積率: 1.3%
人口統計年: 2018
人口順位: 22
人口大きさ: 1 E7
人口値: 6643万5600
人口密度値: 271
GDP統計年元: 2012
GDP値元: 1兆5478億
GDP統計年MER: 2012
GDP順位MER: 6
GDP値MER: 2兆4337億
GDP統計年: 2012
GDP順位: 6
GDP値: 2兆3162億
GDP/人: 36,727
建国形態: 建国
確立形態1: イングランド王国／スコットランド王国（両国とも1707年合同法まで）
確立年月日1: 927年／843年
確立形態2: グレートブリテン王国成立（1707年合同法）
確立年月日2: 1707年5月1日
確立形態3: グレートブリテン及びアイルランド連合王国成立（1800年合同法）
確立年月日3: 1801年1月1日
確立形態4: 現在の国号「グレートブリテン及び北アイルランド連合王国」に変更
確立年月日4: 1927年4月12日
通貨: UKポンド (£)
通貨コード: GBP
時間帯: ±0
夏時間: +1
|ISO 3166-1 = GB / GBR
ccTLD: .uk / .gb
国際電話番号: 44
注記:
'''