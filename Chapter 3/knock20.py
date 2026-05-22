"""
20. JSONデータの読み込み / JSON 数据的读取
Wikipedia記事のJSONファイルを読み込み、「イギリス」に関する記事本文を表示する。
读取 Wikipedia 文章的 JSON 文件，并显示关于"英国（イギリス）"的文章正文。
"""

import json
import gzip
import os


# ファイルパス / 文件路径
FILE_PATH = os.path.join(os.path.dirname(__file__), "jawiki-country.json")


def load_uk_article(path: str = FILE_PATH) -> str:
    """
    JSONファイルから「イギリス」の記事本文を取得する。
    从 JSON 文件中获取"イギリス（英国）"的文章正文。

    ファイルが .gz の場合は gzip で開き、それ以外は通常通り開く。
    如果文件是 .gz 则用 gzip 打开，否则按普通文本打开。
    """
    # 拡張子で開き方を切り替える / 根据扩展名切换打开方式
    opener = gzip.open if path.endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8") as f:
        # 1行 = 1記事の JSON / 一行就是一篇文章的 JSON
        for line in f:
            article = json.loads(line)
            if article.get("title") == "イギリス":
                return article["text"]

    # 見つからなかった場合 / 没找到的情况
    raise ValueError("「イギリス」の記事が見つかりませんでした / 没有找到「イギリス」的文章")


if __name__ == "__main__":
    # 記事本文を表示 / 显示文章正文
    text = load_uk_article()
    print(text)


'''{{redirect|UK}}
{{redirect|英国|春秋時代の諸侯国|英 (春秋)}}
{{Otheruses|ヨーロッパの国|長崎県・熊本県の郷土料理|いぎりす}}
{{基礎情報 国
|略名  =イギリス
|日本語国名 = グレートブリテン及び北アイルランド連合王国
|公式国名 = {{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />
*{{lang|gd|An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath}}（[[スコットランド・ゲール語]]）
*{{lang|cy|Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon}}（[[ウェールズ語]]）
*{{lang|ga|Ríocht Aontaithe na Breataine Móire agus Tuaisceart na hÉireann}}（[[アイルランド語]]）
*{{lang|kw|An Rywvaneth Unys a Vreten Veur hag Iwerdhon Glédh}}（[[コーンウォール語]]）
*{{lang|sco|Unitit Kinrick o Great Breetain an Northren Ireland}}（[[スコットランド語]]）
**{{lang|sco|Claught Kängrick o Docht Brätain an Norlin Airlann}}、{{lang|sco|Unitet Kängdom o Great Brittain an Norlin Airlann}}（アルスター・スコットランド語）</ref>
|国旗画像 = Flag of the United Kingdom.svg
|国章画像 = [[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]
|国章リンク =（[[イギリスの国章|国章]]）
|標語 = {{lang|fr|[[Dieu et mon droit]]}}<br />（[[フランス語]]:[[Dieu et mon droit|神と我が権利]]）
|国歌 = [[女王陛下万歳|{{lang|en|God Save the Queen}}]]{{en icon}}<br />''神よ女王を護り賜え''<br />{{center|[[ファイル:United States Navy Band - God Save the Queen.ogg]]}}
|地図画像 = Europe-UK.svg
|位置画像 = United Kingdom (+overseas territories) in the World (+Antarctica claims).svg
|公用語 = [[英語]]
|首都 = [[ロンドン]]（事実上）
|最大都市 = ロンドン
|元首等肩書 = [[イギリスの君主|女王]]
|元首等氏名 = [[エリザベス2世]]
|首相等肩書 = [[イギリスの首相|首相]]
|首相等氏名 = [[ボリス・ジョンソン]]
|他元首等肩書1 = [[貴族院 (イギリス)|貴族院議長]]
|他元首等氏名1 = [[:en:Norman Fowler, Baron Fowler|ノーマン・ファウラー]]
|他元首等肩書2 = [[庶民院 (イギリス)|庶民院議長]]
|他元首等氏名2 = {{仮リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}
|他元首等肩書3 = [[連合王国最高裁判所|最高裁判所長官]]
|他元首等氏名3 = [[:en:Brenda Hale, Baroness Hale of Richmond|ブレンダ・ヘイル]]
|面積順位 = 76
|面積大きさ = 1 E11'''