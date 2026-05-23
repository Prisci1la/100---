"""
30. 動詞 / 动词
文章 text に含まれる動詞をすべて表示する。
显示文章 text 中包含的所有动词。
"""

from pathlib import Path
from janome.tokenizer import Tokenizer


def load_text(path: str = "text.txt") -> str:
    """テキストファイルを読み込む / 读取文本文件"""
    return Path(__file__).parent.joinpath(path).read_text(encoding="utf-8")


def extract_verbs(text: str) -> list[str]:
    """
    動詞（表層形）をすべて返す。
    返回所有动词（表层形）。
    """
    tokenizer = Tokenizer()
    # 品詞が「動詞」のトークンを抽出 / 抽取词性为「动词」的 token
    return [
        token.surface
        for token in tokenizer.tokenize(text)
        if token.part_of_speech.split(",")[0] == "動詞"
    ]


if __name__ == "__main__":
    text = load_text()
    verbs = extract_verbs(text)
    for v in verbs:
        print(v)
'''
し
除か
なら
し
わから
吹き
遊ん
暮し
来
'''