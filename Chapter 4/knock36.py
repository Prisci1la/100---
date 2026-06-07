"""
36. 単語の出現頻度 / 单词的出现频率
jawiki-country.json をコーパスとして、単語（形態素）の出現頻度を求め、
上位20語を表示する。
将 jawiki-country.json 作为语料库，统计单词（形态素）的出现频率，
并显示出现频率最高的 20 个词。

ファイル形式 / 文件格式：
  1行に1記事の情報（JSON）/ 每行一篇文章（JSON 格式）
  "title"：記事名 / 文章名
  "text" ：記事本文 / 正文
"""

# 导入 json 用于处理 JSON 数据 / json をインポートして JSON データを処理
import json
# 导入 re 用于正则表达式 / re をインポートして正規表現を実施
import re
# 导入 gzip 用于处理压缩文件 / gzip をインポートして圧縮ファイルを処理
import gzip
# 导入 Counter 用于统计频率 / Counter をインポートして頻度を数える
from collections import Counter
# 导入 Path 用于文件处理 / Path をインポートしてファイル処理
from pathlib import Path
# 导入 Janome 分词器 / Janome トークナイザーをインポート
from janome.tokenizer import Tokenizer


# 定义语料库路径
CORPUS = Path(__file__).parent / "jawiki-country.json"


def load_articles() -> list[dict]:  # 加载JSON/JSON.gz文章 / JSON/JSON.gz形式で記事を読み込む
    if CORPUS.suffix == ".gz":
        with gzip.open(CORPUS, "rt", encoding="utf-8") as f:  # 压缩文件 / 圧縮ファイル
            return [json.loads(line) for line in f]
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()]


def remove_markup(text: str) -> str:  # 移除MediaWiki标记 / MediaWikiマークアップを除去
    text = re.sub(r"'{2,5}", "", text)  # 强调 / 強調
    text = re.sub(r"\[\[(?:ファイル|File|Image):[^\]]+\]\]", "", text)  # 文件 / 画像
    text = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", text)  # 内部链接 / 内部リンク
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)  # 外部链接
    text = re.sub(r"\[https?://[^\]]+\]", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)  # 参考标签 / 参考タグ
    text = re.sub(r"<[^>]+/?>", "", text)  # HTML标签 / HTMLタグ
    text = re.sub(r"\{\{[^{}]+\}\}", "", text)  # 模板 / テンプレート
    return text


def tokenize_corpus() -> list[str]:  # 分词统计所有单词 / すべての単語を抽出
    tokenizer = Tokenizer()
    words: list[str] = []
    for article in load_articles():
        clean = remove_markup(article["text"])  # 清理文本 / クリーンアップ
        for tok in tokenizer.tokenize(clean):
            pos = tok.part_of_speech.split(",")[0]  # 获取词性 / 品詞取得
            # 排除符号、空白、助词、助动词 / 記号・空白・助詞・助動詞を除外
            if pos in ("記号", "空白", "助詞", "助動詞") or not tok.surface.strip():
                continue
            words.append(tok.surface)  # 表层形 / 表層形
    return words


def top_words(n: int = 20) -> list[tuple[str, int]]:  # 频率前n个单词 / 頻度上位n個の単語
    return Counter(tokenize_corpus()).most_common(n)  # 统计和排序 / 統計と順序


# 程序入口 / プログラムエントリポイント
if __name__ == "__main__":
    # 遍历频率最高的 20 个单词 / 出現頻度が高い上位 20 個の単語をループ
    for word, freq in top_words(20):
        # 打印单词和频率，以制表符分隔 / 単語と频度をタブ文字で区切って出力
        print(f"{word}\t{freq}")
'''
し      21160
年      19927
|       18125
れ      11617
いる    11398
さ      11031
=       10108
*       9302
===     8504
月      8236
する    7883
人      7467
日      6210
==      6180
国      5912
1       5793
-       5710
||      5587
.       5418
語      4603
'''