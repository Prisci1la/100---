'''
knock56.py: 使用WordSimilarity-353评估 / WordSimilarity-353での評価

比较模型计算的相似度排序与人工评分排序。 / モデルが計算した類似度順位と人手評価の順位を比較する。
输出它们之间的Spearman相关系数。 / それらのSpearman相関係数を表示する。
'''

from chapter6_utils import iter_wordsim353, load_vectors, spearman_correlation  # 导入数据迭代器、模型加载函数和Spearman相关函数 / データ反復器とモデル読み込み関数とSpearman相関関数を導入する


def main():  # 定义主函数 / メイン関数を定義する
    model = load_vectors()  # 读取预训练词向量模型 / 学習済み単語ベクトルモデルを読み込む
    human_scores = []  # 准备保存人工评分 / 人手評価を保存するリストを用意する
    vector_scores = []  # 准备保存模型相似度评分 / モデル類似度スコアを保存するリストを用意する

    for word_a, word_b, human_score in iter_wordsim353():  # 逐行读取WordSimilarity-353数据 / WordSimilarity-353データを1行ずつ読む
        if word_a not in model or word_b not in model:  # 如果某个词不在模型词表中则跳过 / どちらかの単語が語彙に無ければ読み飛ばす
            continue  # 继续处理下一条数据 / 次のデータへ進む
        human_scores.append(human_score)  # 保存人工给出的相似度分数 / 人手の類似度スコアを保存する
        vector_scores.append(model.similarity(word_a, word_b))  # 保存模型计算出的相似度分数 / モデルが計算した類似度スコアを保存する

    correlation = spearman_correlation(human_scores, vector_scores)  # 计算两组分数的Spearman相关系数 / 2組のスコアのSpearman相関係数を計算する
    print(f"used pairs: {len(human_scores)}")  # 输出参与计算的样本对数量 / 計算に使った単語対数を表示する
    print(f"spearman correlation: {correlation:.6f}")  # 输出Spearman相关系数 / Spearman相関係数を表示する


if __name__ == "__main__":  # 只有直接运行文件时才执行 / ファイルを直接実行した場合のみ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
运行结果: / 実行結果:
used pairs: 353
spearman correlation: 0.700017
'''
