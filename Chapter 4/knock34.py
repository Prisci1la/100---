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

# 导入 Janome 分词器 / Janome トークナイザーをインポート
from janome.tokenizer import Tokenizer
# 从 knock30 模块导入 load_text 函数 / knock30 モジュールから load_text 関数をインポート
from knock30 import load_text


def split_sentences(text: str) -> list[str]:  # 按句末符号分割文本 / 文末記号でテキストを文に分割
    sentences: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in ("。", "！", "？", "\n"):  # 句子结尾标志 / 文末記号
            s = buf.strip()
            if s:
                sentences.append(s)
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


def extract_predicates_of_melos(text: str) -> list[str]:  # 提取「メロス」的谓语 / 述語を抽出
    tokenizer = Tokenizer()
    predicates: list[str] = []
    for sent in split_sentences(text):
        tokens = list(tokenizer.tokenize(sent))
        has_melos_subject = any("メロス" in tokens[i].surface and  # 检查「メロス」+ 主格助詞
                                tokens[i + 1].surface in ("は", "が")
                                for i in range(len(tokens) - 1))
        if not has_melos_subject:
            continue
        for token in reversed(tokens):
            pos = token.part_of_speech.split(",")[0]
            if pos in ("動詞", "形容詞"):  # 动词或形容词 / 動詞または形容詞
                predicates.append(token.base_form)
                break
    return predicates


# 程序入口 / プログラムエントリポイント
if __name__ == "__main__":
    # 读取文本内容 / テキスト内容を読み込む
    text = load_text()
    # 遍历谓語并打印 / 述語をループして出力
    for pred in extract_predicates_of_melos(text):
        print(pred)
'''
する
'''