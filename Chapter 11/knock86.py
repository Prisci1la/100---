'''
knock86.py: ミニバッチの作成 / ミニバッチの作成

85で読み込んだ訓練データの一部をpaddingし、BERT用mini-batchを作成する。
/ 85で読み込んだ訓練データの一部をpaddingし、BERT用mini-batchを作成する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from chapter11_utils import DEFAULT_TRAIN_PATH, MODEL_NAME, configure_quiet_mode, create_data_loader, create_dataset, load_tokenizer  # 导入共用工具 / 共通ツールを導入する


def main():  # 定义主函数 / メイン関数を定義する
    configure_quiet_mode()  # 抑制额外日志 / 余計なログを抑制する
    parser = argparse.ArgumentParser(description="knock86: create a BERT mini-batch")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名称 / モデル名
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="SST-2 train.tsv path")  # 训练数据路径 / 訓練データパス
    parser.add_argument("--max-length", type=int, default=128, help="maximum token length")  # 最大token长度 / 最大token長
    parser.add_argument("--batch-size", type=int, default=4, help="mini-batch size")  # batch大小 / batchサイズ
    args = parser.parse_args()  # 解析命令行参数 / コマンドライン引数を解析する

    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読み込む
    dataset = create_dataset(args.train_path, tokenizer, args.max_length, max_examples=args.batch_size)  # 创建小数据集 / 小さなDatasetを作る
    loader = create_data_loader(dataset, tokenizer, batch_size=args.batch_size, shuffle=False)  # 创建DataLoader / DataLoaderを作る
    batch = next(iter(loader))  # 取出一个batch / 1つのbatchを取り出す

    print(f"input_ids shape: {batch['input_ids'].shape}")  # 输出输入形状 / 入力形状を出力する
    print(f"attention_mask shape: {batch['attention_mask'].shape}")  # 输出mask形状 / mask形状を出力する
    print(f"labels: {batch['labels']}")  # 输出标签 / ラベルを出力する
    print(f"texts: {batch['texts']}")  # 输出原文 / 元文を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
Execution result:
input_ids shape: torch.Size([4, 15])
attention_mask shape: torch.Size([4, 15])
labels: tensor([0, 0, 1, 0])
texts: ['hide new secretions from the parental units', 'contains no wit , only labored gags', 'that loves its characters and communicates something rather beautiful about human nature', 'remains utterly satisfied to remain the same throughout']
'''
