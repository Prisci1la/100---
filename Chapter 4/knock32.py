"""
32. 「AのB」/ "A 的 B"
文章 text において、2つの名詞が「の」で連結されている名詞句を
すべて抽出する。
提取文章 text 中两个名词由"の"连接构成的名词短语。

例：「政治の世界」「メロスの友」 など
"""

# 导入 Janome 分词器 / Janome トークナイザーをインポート
from janome.tokenizer import Tokenizer
# 从 knock30 模块导入 load_text 函数 / knock30 モジュールから load_text 関数をインポート
from knock30 import load_text


def extract_a_no_b(text: str) -> list[str]:  # 提取「名词 + の + 名词」短语 / 「名詞 + の + 名詞」を抽出
    tokenizer = Tokenizer()
    tokens = list(tokenizer.tokenize(text))
    phrases: list[str] = []
    for i in range(len(tokens) - 2):
        a, no, b = tokens[i], tokens[i + 1], tokens[i + 2]  # 取出三个连续 token / 3つのトークン
        if (a.part_of_speech.split(",")[0] == "名詞" and  # 第一个词性为名词 / 1つ目は名詞
            no.surface == "の" and  # 第二个为「の」/ 2つ目は「の」
            b.part_of_speech.split(",")[0] == "名詞"):  # 第三个词性为名词 / 3つ目は名詞
            phrases.append(f"{a.surface}の{b.surface}")
    return phrases


# 程序入口 / プログラムエントリポイント
if __name__ == "__main__":
    # 读取文本内容 / テキスト内容を読み込む
    text = load_text()
    # 遍历提取的短语并打印 / 抽出した句をループして出力
    for phrase in extract_a_no_b(text):
        print(phrase)
'''
暴虐の王
村の牧人
'''