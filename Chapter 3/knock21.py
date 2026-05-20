"""
21. カテゴリ名を含む行を抽出 / 提取包含分类名的行
記事中でカテゴリ名を宣言している行を抽出する。
从文章中提取声明分类名的行。

MediaWiki のカテゴリ宣言は [[Category:XXX]] または [[Category:XXX|YYY]] の形式。
MediaWiki 的分类声明形如 [[Category:XXX]] 或 [[Category:XXX|YYY]]。
"""

import re
from knock20 import load_uk_article


def extract_category_lines(text: str) -> list[str]:
    """
    カテゴリ宣言を含む行のリストを返す。
    返回包含分类声明的行的列表。
    """
    # [[Category:...]] を含む行にマッチ / 匹配包含 [[Category:...]] 的行
    pattern = re.compile(r"^.*\[\[Category:.*?\]\].*$", re.MULTILINE)
    return pattern.findall(text)


if __name__ == "__main__":
    # イギリスの本文を読み込み / 读取英国的正文
    text = load_uk_article()

    # カテゴリ行を抽出 / 抽出分类行
    lines = extract_category_lines(text)

    # 1行ずつ表示 / 逐行打印
    for line in lines:
        print(line)


'''
[[Category:イギリス|*]]
[[Category:イギリス連邦加盟国]]
[[Category:英連邦王国|*]]
[[Category:G8加盟国]]
[[Category:欧州連合加盟国|元]]
[[Category:海洋国家]]
[[Category:現存する君主国]]
[[Category:島国]]
[[Category:1801年に成立した国家・領域]]'''