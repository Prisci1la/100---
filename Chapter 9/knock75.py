'''
knock75.py: Padding处理 / Padding処理

实现collate函数，将长度不同的token ID列按长度排序并补齐为同一长度。
/ collate関数を実装し、長さの異なるtoken ID列を長さ順に並べて同じ長さへpaddingする。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from chapter9_utils import (  # 导入第9章共用工具 / 第9章共通ツールを導入する
    DEFAULT_TRAIN_PATH,
    DEFAULT_VECTOR_PATH,
    SSTDataset,
    build_embedding_resources,
    collate_examples,
)


def collate(examples):  # 定义本题要求的collate函数 / 本問で求められるcollate関数を定義する
    return collate_examples(examples)  # 调用共用padding逻辑 / 共通padding処理を呼び出す


def add_common_arguments(parser):  # 添加命令行参数 / コマンドライン引数を追加する
    parser.add_argument("--vector-path", default=str(DEFAULT_VECTOR_PATH), help="word2vec binary/text vector path")  # 词向量路径 / 単語ベクトルパス
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="SST-2 train csv/tsv path")  # 训练数据路径 / 訓練データパス
    parser.add_argument("--max-vocab", type=int, default=50000, help="maximum vectors to load; 0 means all")  # 最大词表数量 / 最大語彙数
    parser.add_argument("--text-vectors", action="store_true", help="read vectors as text format")  # 文本格式词向量开关 / text形式ベクトルの指定
    parser.add_argument("--batch-size", type=int, default=4, help="number of examples to collate")  # 演示batch大小 / 表示用batchサイズ
    return parser  # 返回parser / parserを返す


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock75: collate and pad examples")  # 创建参数解析器 / 引数パーサーを作る
    add_common_arguments(parser)  # 添加参数 / 引数を追加する
    args = parser.parse_args()  # 解析命令行参数 / コマンドライン引数を解析する

    _embedding_matrix, token_to_id, _id_to_token = build_embedding_resources(  # 构建词表 / 語彙を構築する
        vector_path=args.vector_path,  # 指定词向量路径 / 単語ベクトルパスを指定する
        binary=not args.text_vectors,  # 指定词向量格式 / 単語ベクトル形式を指定する
        max_vocab=args.max_vocab,  # 限制词表大小 / 語彙数を制限する
    )
    train_dataset = SSTDataset(args.train_path, token_to_id)  # 创建训练Dataset / 訓練Datasetを作る
    sample_size = min(args.batch_size, len(train_dataset))  # 防止指定数量超过数据集大小 / 指定数がデータセットサイズを超えないようにする
    examples = [train_dataset[index] for index in range(sample_size)]  # 取出前几个样本 / 先頭から数件を取り出す
    batch = collate(examples)  # 执行padding处理 / padding処理を行う

    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 75: Collate Function")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(f"original lengths: {[len(example['input_ids']) for example in examples]}")  # 输出原始长度 / 元の長さを出力する
    print(f"input_ids shape: {batch['input_ids'].shape}")  # 输出输入张量形状 / 入力テンソル形状を出力する
    print(f"label shape: {batch['label'].shape}")  # 输出标签张量形状 / ラベルテンソル形状を出力する
    print(batch)  # 输出batch内容 / batch内容を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

r'''
===== Codex 运行结果 2026-06-28 =====
Command: python knock75.py
Exit code: 0, elapsed: 3.08s
终端输出节选:
==================================================
Knock 75: Collate Function
==================================================
original lengths: [6, 6, 11, 7]
input_ids shape: torch.Size([4, 11])
label shape: torch.Size([4, 1])
{'input_ids': tensor([[    4,  5053,    45,  3305, 31647,   348,   904,  2815,    47,  1276,
          1964],
        [  987, 14528,  4941,   873,    12,   208,   898,     0,     0,     0,
             0],
        [ 5785,    66,    18,    12, 15095,  1594,     0,     0,     0,     0,
             0],
        [ 3475,    87, 15888,    90, 27695, 42637,     0,     0,     0,     0,
             0]]), 'label': tensor([[1.],
        [0.],
        [0.],
        [0.]])}
===== End Codex 运行结果 =====
'''
