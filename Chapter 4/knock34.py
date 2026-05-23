"""
34. 主述の関係 / 主谓关系
文章 text において、「メロス」が主語であるときの述語を抽出する。
提取文章 text 中以"メロス"为主语时的谓语。

注意 / 注意：
  Janome は依存解析できないため、ヒューリスティック規則で判定する。
  Janome 不支持依存解析，使用启发式规则判定。
  規則 / 规则：
    1. 「メロス」+「は/が」を含む文を主語候補とする
    2. 同じ文の最後に出る動詞または形容詞を述語とする
"""

from janome.tokenizer import Tokenizer
from knock30 import load_text


def split_sentences(text: str) -> list[str]:
    """句点で文に分割 / 按句号分割成句子"""
    sentences: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in ("。", "！", "？", "\n"):
            s = buf.strip()
            if s:
                sentences.append(s)
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


def extract_predicates_of_melos(text: str) -> list[str]:
    """
    「メロス」が主語となる文の述語のリストを返す。
    返回以"メロス"为主语的句子的谓语列表。
    """
    tokenizer = Tokenizer()
    predicates: list[str] = []

    for sent in split_sentences(text):
        tokens = list(tokenizer.tokenize(sent))
        # 「メロス」+ 主格助詞（は/が）の組合せがあるかチェック
        # 检查是否存在「メロス」+ 主格助词（は/が）的组合
        has_melos_subject = False
        for i in range(len(tokens) - 1):
            if "メロス" in tokens[i].surface and tokens[i + 1].surface in ("は", "が"):
                has_melos_subject = True
                break
        if not has_melos_subject:
            continue

        # 文末方向から最初に見つかった動詞・形容詞を述語とする
        # 从句末向前找第一个动词或形容词作为谓语
        for token in reversed(tokens):
            pos = token.part_of_speech.split(",")[0]
            if pos in ("動詞", "形容詞"):
                predicates.append(token.base_form)
                break

    return predicates


if __name__ == "__main__":
    text = load_text()
    for pred in extract_predicates_of_melos(text):
        print(pred)
'''
する
'''