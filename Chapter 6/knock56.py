'''
knock56.py: 使用WordSimilarity-353评估 / WordSimilarity-353での評価

比较模型计算的相似度排序与人工评分排序。 / モデルが計算した類似度順位と人手評価の順位を比較する。
输出它们之间的Spearman相关系数。 / それらのSpearman相関係数を表示する。
'''

import csv  # 导入CSV读取模块 / CSV読み込みモジュールを読み込む
import math  # 导入数学计算模块 / 数学計算モジュールを読み込む
from pathlib import Path  # 导入路径处理类 / パス処理クラスを読み込む
from chapter6_utils import load_vectors  # 导入模型加载函数 / モデル読み込み関数を導入する


def iter_wordsim353(path=None):  # 逐行读取WordSimilarity-353数据 / WordSimilarity-353データを1行ずつ読む
    if path is None:
        path = Path(__file__).resolve().parent / "wordsim353" / "combined.csv"
    with path.open(encoding="utf-8") as file:  # 以UTF-8打开CSV文件 / UTF-8でCSVファイルを開く
        reader = csv.DictReader(file)  # 创建按列名读取的CSV解析器 / 列名ベースのCSV読み取り器を作る
        for row in reader:  # 遍历每一行评分数据 / 各行の評価データを順に処理する
            yield row["Word 1"], row["Word 2"], float(row["Human (mean)"])  # 返回两个单词和人工平均分 / 2語と人手平均スコアを返す


def spearman_correlation(xs, ys):  # 计算Spearman秩相关系数 / Spearman順位相関係数を計算する
    def average_ranks(values):  # 计算并列值的平均排名 / 同順位を考慮した平均順位を計算する
        indexed = sorted(enumerate(values), key=lambda item: item[1])  # 按值排序并保留原索引 / 値で並べ替えつつ元の添字を保持する
        ranks = [0.0] * len(values)  # 初始化排名数组 / 順位配列を初期化する
        i = 0  # 初始化起始位置 / 開始位置を初期化する
        while i < len(indexed):  # 处理每一组相同的值 / 同じ値のまとまりごとに処理する
            j = i + 1  # 从下一位开始查找 / 次の位置から探索を始める
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:  # 找到所有并列元素 / 同じ値が続く範囲を見つける
                j += 1  # 扩展并列区间 / 同順位区間を広げる
            rank = (i + 1 + j) / 2  # 计算平均排名 / 平均順位を計算する
            for k in range(i, j):  # 给并列元素赋相同平均排名 / 同順位要素へ同じ平均順位を割り当てる
                ranks[indexed[k][0]] = rank  # 写回对应原索引位置 / 元の添字位置へ書き戻す
            i = j  # 继续处理下一组值 / 次の値のグループへ進む
        return ranks  # 返回排名数组 / 順位配列を返す

    rx = average_ranks(xs)  # 计算第一组数据的排名 / 1つ目のデータの順位を計算する
    ry = average_ranks(ys)  # 计算第二组数据的排名 / 2つ目のデータの順位を計算する
    mean_x = sum(rx) / len(rx)  # 计算第一组排名均值 / 1つ目の順位平均を計算する
    mean_y = sum(ry) / len(ry)  # 计算第二组排名均值 / 2つ目の順位平均を計算する
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry))  # 计算协方差分子 / 共分散の分子を計算する
    var_x = sum((x - mean_x) ** 2 for x in rx)  # 计算第一组方差项 / 1つ目の分散項を計算する
    var_y = sum((y - mean_y) ** 2 for y in ry)  # 计算第二组方差项 / 2つ目の分散項を計算する
    return cov / math.sqrt(var_x * var_y)  # 返回Spearman相关系数 / Spearman相関係数を返す


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
