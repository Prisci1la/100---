'''
knock52.py: 输出相似度最高的10个单词 / 類似度の高い10語の出力

寻找与“United_States”最接近的单词。 / 「United_States」に最も近い単語を探す。
输出前10个单词及其相似度。 / 上位10語とその類似度を表示する。
'''

from chapter6_utils import load_vectors  # 导入模型加载函数 / モデル読み込み関数を導入する


def main():  # 定义主函数 / メイン関数を定義する
    model = load_vectors()  # 读取预训练词向量模型 / 学習済み単語ベクトルモデルを読み込む
    for word, similarity in model.most_similar("United_States", topn=10):  # 查询与United_States最相近的前10个单词 / United_Statesに最も近い上位10語を検索する
        print(f"{word}\t{similarity:.6f}")  # 输出单词和相似度 / 単語と類似度を表示する


if __name__ == "__main__":  # 只有直接运行文件时才执行 / ファイルを直接実行した場合のみ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
运行结果: / 実行結果:
Unites_States	0.787725
Untied_States	0.754137
United_Sates	0.740072
U.S.	0.731077
theUnited_States	0.640439
America	0.617841
UnitedStates	0.616731
Europe	0.613299
countries	0.604480
Canada	0.601907
'''
