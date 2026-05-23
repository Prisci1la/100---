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

import json
import re
import gzip
from collections import Counter
from pathlib import Path
from janome.tokenizer import Tokenizer


CORPUS = Path(__file__).parent / "jawiki-country.json"


def load_articles() -> list[dict]:
    """
    JSON / JSON.gz の両方に対応して記事リストを読み込む。
    同时兼容 JSON / JSON.gz，返回文章列表。
    """
    if CORPUS.suffix == ".gz":
        with gzip.open(CORPUS, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()]


def remove_markup(text: str) -> str:
    """
    第3章を参考に MediaWiki マークアップを簡易除去する。
    参考第 3 章，简单移除 MediaWiki 标记。
    """
    text = re.sub(r"'{2,5}", "", text)                              # 強調 / 强调
    text = re.sub(r"\[\[(?:ファイル|File|Image):[^\]]+\]\]", "", text)  # ファイル / 文件
    text = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", text)    # 内部リンク / 内部链接
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text) # 外部リンク文字付き
    text = re.sub(r"\[https?://[^\]]+\]", "", text)                # 外部リンク
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+/?>", "", text)
    text = re.sub(r"\{\{[^{}]+\}\}", "", text)                     # テンプレート
    return text


def tokenize_corpus() -> list[str]:
    """全記事を形態素に分割した全トークンを返す。/ 返回全部文章分词后的全部 token。"""
    tokenizer = Tokenizer()
    tokens: list[str] = []
    for article in load_articles():
        clean = remove_markup(article["text"])
        for tok in tokenizer.tokenize(clean):
            pos = tok.part_of_speech.split(",")[0]
            # 記号と空白は除外 / 排除符号和空白
            if pos == "記号" or not tok.surface.strip():
                continue
            tokens.append(tok.surface)
    return tokens


def top_words(n: int = 20) -> list[tuple[str, int]]:
    """頻度上位 n 語を返す。/ 返回频率前 n 个词。"""
    return Counter(tokenize_corpus()).most_common(n)


if __name__ == "__main__":
    for word, freq in top_words(20):
        print(f"{word}\t{freq}")
'''
の      72511
は      42775
に      40678
が      36768
を      31621
た      29023
で      28132
と      22378
て      21603
し      21183
年      19927
|       18125
れ      11618
いる    11398
さ      11031
=       10108
ある    9875
*       9302
も      8871
===     8504
'''