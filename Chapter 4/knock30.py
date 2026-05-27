"""
30. 動詞 / 动词
文章 text に含まれる動詞をすべて表示する。
显示文章 text 中包含的所有动词。
"""

# 导入 Path 用于处理文件路径 / Path をインポートしてファイルパスを処理
from pathlib import Path
# 导入 Janome 分词器进行日文分词 / Janome トークナイザーをインポートして日本語を分かち書き
from janome.tokenizer import Tokenizer


# 定义函数：读取文本文件 / 関数を定義：テキストファイルを読み込む
def load_text(path: str = "text.txt") -> str:
    """テキストファイルを読み込む / 读取文本文件"""
    # 从当前文件目录读取指定路径的文件，使用 UTF-8 编码 / 現在のファイルディレクトリから指定されたパスのファイルを読み込み、UTF-8 エンコーディングを使用
    return Path(__file__).parent.joinpath(path).read_text(encoding="utf-8")


def extract_verbs(text: str) -> list[str]:  # 提取所有动词 / すべての動詞を抽出
    tokenizer = Tokenizer()
    return [
        token.surface
        for token in tokenizer.tokenize(text)
        if token.part_of_speech.split(",")[0] == "動詞"  # 词性为「动词」/ 品詞が「動詞」
    ]


# 程序入口 / プログラムエントリポイント
if __name__ == "__main__":
    # 读取文本内容 / テキスト内容を読み込む
    text = load_text()
    # 提取所有动词 / すべての動詞を抽出
    verbs = extract_verbs(text)
    # 逐个打印每个动词 / 各動詞を1つずつ出力
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