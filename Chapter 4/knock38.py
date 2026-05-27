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

# 导入 math 用于数学计算 / math をインポートして数学的計算を実施
import math
# 导入 Counter 用于统计频率 / Counter をインポートして頻度を数える
from collections import Counter
# 导入 Janome 分词器 / Janome トークナイザーをインポート
from janome.tokenizer import Tokenizer

# 从 knock36 模块导入函数 / knock36 モジュールから関数をインポート
from knock36 import load_articles, remove_markup


def _extract_nouns(tokenizer: Tokenizer, text: str) -> list[str]:  # 提取名词 / 名詞を抽出
    nouns: list[str] = []
    for tok in tokenizer.tokenize(text):
        features = tok.part_of_speech.split(",")
        if features[0] == "名詞" and features[1] not in ("非自立", "代名詞", "数"):  # 名詞かつ排除対象外
            nouns.append(tok.surface)
    return nouns


def compute_tfidf(target_title: str = "日本") -> list[tuple[str, float, float, float]]:  # 计算TF-IDF / TF-IDF計算
    tokenizer = Tokenizer()
    articles = load_articles()
    doc_nouns: list[set[str]] = []
    target_tokens: list[str] = []
    for article in articles:
        clean = remove_markup(article["text"])
        nouns = _extract_nouns(tokenizer, clean)  # 提取名词 / 名詞抽出
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
        tf = count / total  # TF = 出现次数 / 总数 / TF=出現回数/総数
        df = sum(1 for s in doc_nouns if word in s)  # 含有词的文档数
        idf = math.log(n_docs / df) if df > 0 else 0.0  # IDF = log(总数/含有数)
        results.append((word, tf, idf, tf * idf))
    results.sort(key=lambda x: x[3], reverse=True)  # 按TF*IDF降序 / 降順
    return results


# 程序入口 / プログラムエントリポイント
if __name__ == "__main__":
    # 获取前 20 个 TF-IDF 值最高的单词 / TF-IDF が高い上位 20 語を取得
    top = compute_tfidf("日本")[:20]
    # 打印表头 / ヘッダーを出力
    print(f"{'word':<15}{'TF':>10}{'IDF':>10}{'TF*IDF':>12}")
    # 遍历前 20 个结果 / 上位 20 件の結果をループ
    for word, tf, idf, tfidf in top:
        # 打印单词及其 TF、IDF、TF*IDF 值 / 単語と TF・IDF・TF*IDF を出力
        print(f"{word:<15}{tf:>10.5f}{idf:>10.5f}{tfidf:>12.5f}")
