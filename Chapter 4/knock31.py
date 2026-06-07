"""
31. 動詞の原型 / 动词的原型
文章 text に含まれる動詞と、その原型をすべて表示する。
显示文章 text 中所有的动词及其原型（基本形）。
"""

# 导入 Janome 分词器 / Janome トークナイザーをインポート
from janome.tokenizer import Tokenizer
# 从 knock30 模块导入 load_text 函数 / knock30 モジュールから load_text 関数をインポート
from knock30 import load_text


def extract_verbs_with_lemma(text: str) -> list[tuple[str, str]]:  # 提取动词和原形 / 動詞とその原形を抽出
    tokenizer = Tokenizer()
    results: list[tuple[str, str]] = []
    for token in tokenizer.tokenize(text):
        if token.part_of_speech.split(",")[0] == "動詞":  # 词性为「动词」/ 品詞が「動詞」
            results.append((token.surface, token.base_form))  # (表层形, 原形) 元组 / (表層形, 原形)
    return results


# 程序入口 / プログラムエントリポイント
if __name__ == "__main__":
    # 读取文本内容 / テキスト内容を読み込む
    text = load_text()
    # 遍历动词及其原形，并打印 / 动詞とその原形をループして出力
    for surface, lemma in extract_verbs_with_lemma(text):
        # 以制表符分隔打印表层形和原形 / 表層形と原形をタブ文字で区切って出力
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