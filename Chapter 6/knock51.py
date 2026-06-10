'''
knock51.py: 单词相似度 / 単語の類似度

比较"United_States"和"U.S."的词向量。 / 「United_States」と「U.S.」の単語ベクトルを比較する。
计算并输出两者的余弦相似度。 / 2語のコサイン類似度を計算して表示する。
'''

import math  # 导入数学计算模块 / 数学計算モジュールを読み込む
from chapter6_utils import load_vectors  # 导入词向量模型加载函数 / 単語ベクトル読み込み関数を導入する


def cosine_similarity(vec_a, vec_b):  # 计算两个向量的余弦相似度 / 2つのベクトルのコサイン類似度を計算する
    dot = float(vec_a @ vec_b)  # 计算点积 / 内積を計算する
    norm_a = math.sqrt(float(vec_a @ vec_a))  # 计算第一个向量的范数 / 1つ目のベクトルのノルムを計算する
    norm_b = math.sqrt(float(vec_b @ vec_b))  # 计算第二个向量的范数 / 2つ目のベクトルのノルムを計算する
    return dot / (norm_a * norm_b)  # 返回余弦相似度 / コサイン類似度を返す


def main():  # 定义主函数 / メイン関数を定義する
    model = load_vectors()  # 读取预训练词向量模型 / 学習済み単語ベクトルモデルを読み込む
    similarity = cosine_similarity(model["United_States"], model["U.S."])  # 计算两个国家写法之间的余弦相似度 / 2つの国名表記間のコサイン類似度を計算する
    print(f"cosine_similarity(United_States, U.S.) = {similarity:.6f}")  # 输出相似度结果 / 類似度結果を表示する


if __name__ == "__main__":  # 只有直接运行文件时才执行 / ファイルを直接実行した場合のみ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
运行结果: / 実行結果:
cosine_similarity(United_States, U.S.) = 0.731077
'''
