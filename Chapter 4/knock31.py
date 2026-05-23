"""
31. 動詞の原型 / 动词的原型
文章 text に含まれる動詞と、その原型をすべて表示する。
显示文章 text 中所有的动词及其原型（基本形）。
"""

from janome.tokenizer import Tokenizer
from knock30 import load_text


def extract_verbs_with_lemma(text: str) -> list[tuple[str, str]]:
    """
    (表層形, 原型) のリストを返す。
    返回 (表层形, 原型) 的列表。
    """
    tokenizer = Tokenizer()
    results: list[tuple[str, str]] = []
    for token in tokenizer.tokenize(text):
        if token.part_of_speech.split(",")[0] == "動詞":
            # token.base_form は原型（基本形）/ token.base_form 是原型（基本形）
            results.append((token.surface, token.base_form))
    return results


if __name__ == "__main__":
    text = load_text()
    for surface, lemma in extract_verbs_with_lemma(text):
        print(f"{surface}\t{lemma}")
'''
し      する
除か    除く
なら    なる
し      する
わから  わかる
吹き    吹く
遊ん    遊ぶ
暮し    暮す
来      来る
'''