"""
32. 「AのB」/ "A 的 B"
文章 text において、2つの名詞が「の」で連結されている名詞句を
すべて抽出する。
提取文章 text 中两个名词由"の"连接构成的名词短语。

例：「政治の世界」「メロスの友」 など
"""

from janome.tokenizer import Tokenizer
from knock30 import load_text


def extract_a_no_b(text: str) -> list[str]:
    """
    「名詞 + の + 名詞」のパターンを抽出する。
    抽取"名词 + の + 名词"模式。
    """
    tokenizer = Tokenizer()
    tokens = list(tokenizer.tokenize(text))
    phrases: list[str] = []
    # 3トークンの窓でスキャン / 用 3 个 token 的窗口扫描
    for i in range(len(tokens) - 2):
        a, no, b = tokens[i], tokens[i + 1], tokens[i + 2]
        if (
            a.part_of_speech.split(",")[0] == "名詞"
            and no.surface == "の"
            and b.part_of_speech.split(",")[0] == "名詞"
        ):
            phrases.append(f"{a.surface}の{b.surface}")
    return phrases


if __name__ == "__main__":
    text = load_text()
    for phrase in extract_a_no_b(text):
        print(phrase)
'''
暴虐の王
村の牧人
'''