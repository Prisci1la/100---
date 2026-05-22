"""
37. 名詞の出現頻度 / 名词的出现频率
コーパスにおける名詞の出現頻度を求め、上位20語を表示する。
统计语料库中名词的出现频率，并显示出现频率最高的 20 个词。
"""

from collections import Counter
from janome.tokenizer import Tokenizer

from knock36 import load_articles, remove_markup


def tokenize_nouns() -> list[str]:
    """全記事から名詞だけを抽出する。/ 从所有文章中只抽取名词。"""
    tokenizer = Tokenizer()
    nouns: list[str] = []
    for article in load_articles():
        clean = remove_markup(article["text"])
        for tok in tokenizer.tokenize(clean):
            features = tok.part_of_speech.split(",")
            # 「名詞」かつ非自立・代名詞・数を除外
            # 「名词」且排除「非自立 / 代名词 / 数」子类
            if features[0] == "名詞" and features[1] not in ("非自立", "代名詞", "数"):
                nouns.append(tok.surface)
    return nouns


def top_nouns(n: int = 20) -> list[tuple[str, int]]:
    """頻度上位 n 個の名詞を返す。/ 返回频率前 n 个名词。"""
    return Counter(tokenize_nouns()).most_common(n)


if __name__ == "__main__":
    for noun, freq in top_nouns(20):
        print(f"{noun}\t{freq}")
