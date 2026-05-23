"""
38. TF・IDF
日本に関する記事における名詞の TF・IDF スコアを求め、
上位20語と TF / IDF / TF・IDF を表示する。
计算关于"日本"的文章中名词的 TF·IDF 分数，
显示前 20 个词及其 TF, IDF, TF·IDF。

定義 / 定义：
  TF(w, d) = 文書 d における w の出現回数 / 文書 d の総単語数
  IDF(w)   = log(全文書数 / w を含む文書数)
  TF・IDF  = TF × IDF
"""

import math
from collections import Counter
from janome.tokenizer import Tokenizer

from knock36 import load_articles, remove_markup


def _extract_nouns(tokenizer: Tokenizer, text: str) -> list[str]:
    """テキストから名詞を抽出 / 从文本中抽取名词"""
    nouns: list[str] = []
    for tok in tokenizer.tokenize(text):
        features = tok.part_of_speech.split(",")
        if features[0] == "名詞" and features[1] not in ("非自立", "代名詞", "数"):
            nouns.append(tok.surface)
    return nouns


def compute_tfidf(target_title: str = "日本") -> list[tuple[str, float, float, float]]:
    """
    target_title の記事における名詞の TF・IDF を計算する。
    返回值 / 返回: (word, tf, idf, tfidf) のリスト
    """
    tokenizer = Tokenizer()
    articles = load_articles()

    # 全記事の名詞集合を作る（IDF 用）/ 构建每篇文章的名词集合（用于 IDF）
    doc_nouns: list[set[str]] = []
    target_tokens: list[str] = []

    for article in articles:
        clean = remove_markup(article["text"])
        nouns = _extract_nouns(tokenizer, clean)
        doc_nouns.append(set(nouns))
        if article["title"] == target_title:
            target_tokens = nouns

    if not target_tokens:
        raise ValueError(f"記事 / 文章未找到: {target_title}")

    n_docs = len(articles)
    tf_counter = Counter(target_tokens)
    total = sum(tf_counter.values())

    results: list[tuple[str, float, float, float]] = []
    for word, count in tf_counter.items():
        tf = count / total
        df = sum(1 for s in doc_nouns if word in s)  # word を含む記事数
        idf = math.log(n_docs / df) if df > 0 else 0.0
        results.append((word, tf, idf, tf * idf))

    results.sort(key=lambda x: x[3], reverse=True)
    return results


if __name__ == "__main__":
    top = compute_tfidf("日本")[:20]
    print(f"{'word':<15}{'TF':>10}{'IDF':>10}{'TF*IDF':>12}")
    for word, tf, idf, tfidf in top:
        print(f"{word:<15}{tf:>10.5f}{idf:>10.5f}{tfidf:>12.5f}")
'''
word                   TF       IDF      TF*IDF
天皇                0.00242   2.80538     0.00679
*「                0.00163   4.12713     0.00672
::                0.00172   3.90399     0.00672
倭国                0.00130   4.82028     0.00628
倭                 0.00126   4.12713     0.00518
琉球                0.00116   4.41482     0.00513
朝鮮                0.00191   2.68022     0.00511
沖縄                0.00172   2.94848     0.00507
列島                0.00163   3.11553     0.00507
県                 0.00707   0.69315     0.00490
北海道               0.00158   2.62306     0.00415
昭和                0.00144   2.68022     0.00386
日本書紀              0.00070   5.51343     0.00385
明治                0.00149   2.56899     0.00382
韓国                0.00214   1.72924     0.00370
中国                0.00465   0.79493     0.00370
台湾                0.00233   1.56219     0.00363
自衛隊               0.00135   2.68022     0.00362
南西諸島              0.00074   4.82028     0.00359
アイヌ               0.00074   4.82028     0.00359
'''