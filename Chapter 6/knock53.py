'''
knock53.py: 基于加法构成性的类比 / 加法構成性によるアナロジー

计算 vec("Spain") - vec("Madrid") + vec("Athens")。 / vec("Spain") - vec("Madrid") + vec("Athens") を計算する。
输出与结果向量最接近的10个单词。 / 結果ベクトルに最も近い10語を表示する。
'''

from chapter6_utils import load_vectors  # 导入模型加载函数 / モデル読み込み関数を導入する


def main():  # 定义主函数 / メイン関数を定義する
    model = load_vectors()  # 读取预训练词向量模型 / 学習済み単語ベクトルモデルを読み込む
    results = model.most_similar(  # 执行向量类比查询 / ベクトルアナロジー検索を実行する
        positive=["Spain", "Athens"],  # 指定正向向量部分 / 正のベクトル成分を指定する
        negative=["Madrid"],  # 指定负向向量部分 / 負のベクトル成分を指定する
        topn=10,  # 只返回前10个结果 / 上位10件だけを返す
    )

    for word, similarity in results:  # 遍历类比结果 / アナロジー結果を順に処理する
        print(f"{word}\t{similarity:.6f}")  # 输出单词和相似度 / 単語と類似度を表示する


if __name__ == "__main__":  # 只有直接运行文件时才执行 / ファイルを直接実行した場合のみ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
运行结果: / 実行結果:
Greece	0.689848
Aristeidis_Grigoriadis	0.560685
Ioannis_Drymonakos	0.555291
Greeks	0.545069
Ioannis_Christou	0.540086
Hrysopiyi_Devetzi	0.524844
Heraklio	0.520776
Athens_Greece	0.516881
Lithuania	0.516687
Iraklion	0.514679
'''
