'''
knock70.py: 单词嵌入的读取 / 単語埋め込みの読み込み

读取预训练词向量，创建第0行为PAD零向量的embedding矩阵。
/ 学習済み単語ベクトルを読み、0行目をPADのゼロベクトルにしたembedding行列を作る。
'''

from chapter8_utils import build_embedding_resources, parse_common_args  # 导入embedding构建函数 / embedding構築関数を導入する


def main():  # 定义主函数 / メイン関数を定義する
    args = parse_common_args("knock70: build embedding matrix")  # 解析命令行参数 / コマンドライン引数を解析する
    embedding_matrix, token_to_id, id_to_token = build_embedding_resources(  # 构建embedding资源 / embeddingリソースを構築する
        vector_path=args.vector_path,  # 指定词向量路径 / 単語ベクトルパスを指定する
        binary=not args.text_vectors,  # 指定binary或text格式 / binaryまたはtext形式を指定する
        max_vocab=args.max_vocab,  # 限制读取词表大小 / 読み込む語彙数を制限する
    )

    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 70: Embedding Matrix")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(f"shape: {embedding_matrix.shape}")  # 输出矩阵形状 / 行列の形状を出力する
    print(f"PAD id: {token_to_id['<PAD>']}")  # 输出PAD的ID / PADのIDを出力する
    print(f"first vector sum: {embedding_matrix[0].sum():.1f}")  # 确认第0行是零向量 / 0行目がゼロベクトルであることを確認する
    print(f"first tokens: {id_to_token[:10]}")  # 输出前10个token / 先頭10個のtokenを出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す
