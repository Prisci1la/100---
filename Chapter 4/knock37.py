"""
37. 名詞の出現頻度 / 名词的出现频率
コーパスにおける名詞の出現頻度を求め、上位20語を表示する。
统计语料库中名词的出现频率，并显示出现频率最高的 20 个词。
"""

# 导入 Counter 用于统计频率 / Counter をインポートして頻度を数える
from collections import Counter
# 导入 Janome 分词器 / Janome トークナイザーをインポート
from janome.tokenizer import Tokenizer

# 从 knock36 模块导入函数 / knock36 モジュールから関数をインポート
from knock36 import load_articles, remove_markup


def tokenize_nouns() -> list[str]:  # 从所有文章中提取名词 / すべての記事から名詞を抽出
    tokenizer = Tokenizer()
    nouns: list[str] = []
    for article in load_articles():
        clean = remove_markup(article["text"])
        for tok in tokenizer.tokenize(clean):
            features = tok.part_of_speech.split(",")
            if features[0] == "名詞" and features[1] not in ("非自立", "代名詞", "数"):  # 名詞かつ排除対象外
                nouns.append(tok.surface)
    return nouns


def top_nouns(n: int = 20) -> list[tuple[str, int]]:  # 频率前n个名词 / 出現頻度高い名詞
    return Counter(tokenize_nouns()).most_common(n)  # 统计和排序 / 統計と順序


# 程序入口 / プログラムエントリポイント
if __name__ == "__main__":
    # 遍历频率最高的 20 个名词 / 出現頻度が高い上位 20 個の名詞をループ
    for noun, freq in top_nouns(20):
        # 打印名詞和频率，以制表符分隔 / 名詞と频度をタブ文字で区切って出力
        print(f"{noun}\t{freq}")
'''
年      19927
|       18125
=       10108
*       9302
===     8504
月      8236
人      7467
==      6180
国      5912
-       5710
日      5611
||      5587
.       5418
語      4603
的      4240
,       3840
:       3822
日本    3078
|-      3064
政府    3005
'''