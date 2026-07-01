'''
knock77.py: GPU上的学习 / GPU上での学習

把模型和mini-batch移动到GPU，在GPU可用时使用CUDA训练BoW模型。
/ モデルとmini-batchをGPUへ移し、CUDAが使える場合はGPUでBoWモデルを学習する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # 导入PyTorch / PyTorchを導入する

from chapter9_utils import (  # 导入第9章共用工具 / 第9章共通ツールを導入する
    DEFAULT_DEV_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VECTOR_PATH,
    BoWClassifier,
    SSTDataset,
    build_embedding_resources,
    create_data_loader,
    evaluate_accuracy,
    get_device,
    train_one_epoch_batched,
)


def add_common_arguments(parser):  # 添加命令行参数 / コマンドライン引数を追加する
    parser.add_argument("--vector-path", default=str(DEFAULT_VECTOR_PATH), help="word2vec binary/text vector path")  # 词向量路径 / 単語ベクトルパス
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="SST-2 train csv/tsv path")  # 训练数据路径 / 訓練データパス
    parser.add_argument("--dev-path", default=str(DEFAULT_DEV_PATH), help="SST-2 dev csv/tsv path")  # 开发数据路径 / 開発データパス
    parser.add_argument("--max-vocab", type=int, default=50000, help="maximum vectors to load; 0 means all")  # 最大词表数量 / 最大語彙数
    parser.add_argument("--text-vectors", action="store_true", help="read vectors as text format")  # 文本格式词向量开关 / text形式ベクトルの指定
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")  # 训练轮数 / 学習epoch数
    parser.add_argument("--batch-size", type=int, default=256, help="mini-batch size")  # GPU用batch大小 / GPU用batchサイズ
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")  # 学习率 / 学習率
    return parser  # 返回parser / parserを返す


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock77: train BoW classifier on GPU")  # 创建参数解析器 / 引数パーサーを作る
    add_common_arguments(parser)  # 添加参数 / 引数を追加する
    args = parser.parse_args()  # 解析命令行参数 / コマンドライン引数を解析する

    device = get_device()  # 优先取得CUDA设备 / CUDAデバイスを優先して取得する
    embedding_matrix, token_to_id, _id_to_token = build_embedding_resources(  # 构建embedding资源 / embeddingリソースを構築する
        vector_path=args.vector_path,  # 指定词向量路径 / 単語ベクトルパスを指定する
        binary=not args.text_vectors,  # 指定词向量格式 / 単語ベクトル形式を指定する
        max_vocab=args.max_vocab,  # 限制词表大小 / 語彙数を制限する
    )
    train_dataset = SSTDataset(args.train_path, token_to_id)  # 创建训练Dataset / 訓練Datasetを作る
    dev_dataset = SSTDataset(args.dev_path, token_to_id)  # 创建开发Dataset / 開発Datasetを作る
    train_loader = create_data_loader(train_dataset, batch_size=args.batch_size, shuffle=True)  # 创建训练DataLoader / 訓練DataLoaderを作る
    dev_loader = create_data_loader(dev_dataset, batch_size=args.batch_size, shuffle=False)  # 创建开发DataLoader / 開発DataLoaderを作る
    model = BoWClassifier(embedding_matrix, freeze_embeddings=True).to(device)  # 创建模型并移动到设备 / モデルを作ってデバイスへ移す
    loss_fn = torch.nn.BCEWithLogitsLoss()  # 创建损失函数 / 損失関数を作る
    optimizer = torch.optim.Adam(model.linear.parameters(), lr=args.lr)  # 只优化线性分类层 / 線形分類層だけを最適化する

    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 77: GPU Training")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(f"device: {device}")  # 输出设备 / デバイスを出力する
    print(f"batch size: {args.batch_size}")  # 输出batch大小 / batchサイズを出力する

    for epoch in range(1, args.epochs + 1):  # 按epoch循环 / epochごとに繰り返す
        train_loss = train_one_epoch_batched(model, train_loader, optimizer, loss_fn, device)  # GPU上训练一轮 / GPU上で1epoch学習する
        dev_accuracy = evaluate_accuracy(model, dev_loader, device)  # GPU上计算开发集正解率 / GPU上で開発セット正解率を計算する
        print(f"epoch {epoch:02d}: train_loss={train_loss:.6f}, dev_accuracy={dev_accuracy:.6f}")  # 输出进度 / 進捗を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

