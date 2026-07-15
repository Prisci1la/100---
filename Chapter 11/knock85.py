'''
knock85.py: データセットの準備 / データセットの準備

SST-2のtrain/devを読み込み、すべてのテキストをBERT token列に変換する。
/ SST-2のtrain/devを読み込み、すべてのテキストをBERT token列に変換する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from chapter11_utils import DEFAULT_DEV_PATH, DEFAULT_TRAIN_PATH, MODEL_NAME, configure_quiet_mode, create_dataset, load_tokenizer  # 导入共用工具 / 共通ツールを導入する


def main():  # 定义主函数 / メイン関数を定義する
    configure_quiet_mode()  # 抑制额外日志 / 余計なログを抑制する
    parser = argparse.ArgumentParser(description="knock85: prepare SST-2 for BERT")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名称 / モデル名
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="SST-2 train.tsv path")  # 训练数据路径 / 訓練データパス
    parser.add_argument("--dev-path", default=str(DEFAULT_DEV_PATH), help="SST-2 dev.tsv path")  # 开发数据路径 / 開発データパス
    parser.add_argument("--max-length", type=int, default=128, help="maximum token length")  # 最大token长度 / 最大token長
    parser.add_argument("--max-examples", type=int, default=None, help="limit examples for quick check")  # 快速检查用上限 / 確認用上限
    args = parser.parse_args()  # 解析命令行参数 / コマンドライン引数を解析する

    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読み込む
    train_dataset = create_dataset(args.train_path, tokenizer, args.max_length, args.max_examples)  # 创建训练Dataset / 訓練Datasetを作る
    dev_dataset = create_dataset(args.dev_path, tokenizer, args.max_length, args.max_examples)  # 创建开发Dataset / 開発Datasetを作る
    first = train_dataset[0]  # 取出第一个样本 / 最初のサンプルを取り出す
    tokens = tokenizer.convert_ids_to_tokens(first["input_ids"])  # ID转换为token / IDをtokenへ変換する

    print(f"train examples: {len(train_dataset)}")  # 输出训练样本数 / 訓練サンプル数を出力する
    print(f"dev examples: {len(dev_dataset)}")  # 输出开发样本数 / 開発サンプル数を出力する
    print(f"text: {first['text']}")  # 输出原文 / 元文を出力する
    print(f"label: {first['labels']}")  # 输出标签 / ラベルを出力する
    print(f"tokens: {tokens}")  # 输出token列 / token列を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
Execution result:
train examples: 67349
dev examples: 872
text: hide new secretions from the parental units
label: 0
tokens: ['[CLS]', 'hide', 'new', 'secret', '##ions', 'from', 'the', 'parental', 'units', '[SEP]']
'''
