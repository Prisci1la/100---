'''
knock71.py: 数据集的读取 / データセットの読み込み

使用自定义Dataset读取SST-2，并将文本转换为token ID列。
/ カスタムDatasetでSST-2を読み、テキストをtoken ID列へ変換する。
'''

from chapter8_utils import SSTDataset, build_embedding_resources, parse_common_args  # 导入第8章工具 / 第8章ツールを導入する


def main():  # 定义主函数 / メイン関数を定義する
    args = parse_common_args("knock71: load SST-2 dataset")  # 解析命令行参数 / コマンドライン引数を解析する
    _embedding_matrix, token_to_id, _id_to_token = build_embedding_resources(  # 读取词向量并创建词表 / 単語ベクトルを読んで語彙を作る
        vector_path=args.vector_path,  # 指定词向量路径 / 単語ベクトルパスを指定する
        binary=not args.text_vectors,  # 指定词向量格式 / 単語ベクトル形式を指定する
        max_vocab=args.max_vocab,  # 限制词表大小 / 語彙数を制限する
    )
    train_dataset = SSTDataset(args.train_path, token_to_id)  # 创建训练Dataset / 訓練Datasetを作る
    dev_dataset = SSTDataset(args.dev_path, token_to_id)  # 创建开发Dataset / 開発Datasetを作る

    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 71: SST-2 Dataset")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(f"train examples: {len(train_dataset)}")  # 输出训练样本数 / 訓練サンプル数を出力する
    print(f"dev examples: {len(dev_dataset)}")  # 输出开发样本数 / 開発サンプル数を出力する
    print(train_dataset[0])  # 输出第一个样本 / 先頭サンプルを出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す
