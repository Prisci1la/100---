"""
22. カテゴリ名の抽出 / 提取分类名
記事のカテゴリ名を（行単位ではなく名前で）抽出する。
提取文章的分类名（不是整行，而是仅名称本身）。

例: [[Category:イギリス|*]] → 「イギリス」
例: [[Category:イギリス|*]] → "イギリス"
"""

import re
from knock20 import load_uk_article


def extract_category_names(text: str) -> list[str]:
    """
    カテゴリ名のみのリストを返す（パイプ以降のソートキーは除外）。
    返回仅分类名的列表（去掉竖线后的排序键）。
    """
    # キャプチャグループでカテゴリ名部分のみ取得
    # 用捕获组只取分类名部分
    #   [[Category:    ← 固定
    #   ([^|\]]+)     ← カテゴリ名 / 分类名（不含 | 和 ]）
    #   (?:\|[^\]]*)? ← 任意のソートキー / 可选的排序键
    #   ]]            ← 固定
    pattern = re.compile(r"\[\[Category:([^|\]]+)(?:\|[^\]]*)?\]\]")
    return pattern.findall(text)


if __name__ == "__main__":
    # 本文取得 / 取得正文
    text = load_uk_article()

    # カテゴリ名抽出 / 提取分类名
    names = extract_category_names(text)

    # 結果表示 / 显示结果
    for name in names:
        print(name)

'''
イギリス
イギリス連邦加盟国
英連邦王国
G8加盟国
欧州連合加盟国
海洋国家
現存する君主国
島国
1801年に成立した国家・領域
'''