'''
knock96.py: プロンプトによる感情分析 / プロンプトによる感情分析

ファインチューニングなしで、promptによるSST-2開発データの正解率を計測する。
/ ファインチューニングなしで、promptによるSST-2開発データの正解率を計測する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from tqdm.auto import tqdm  # 导入进度条 / 進捗バーを導入する

from chapter12_utils import DEFAULT_DEV_PATH, MODEL_NAME, get_device, load_causal_lm, load_tokenizer, prompt_sentiment_predict, read_sst2_rows  # 导入共用工具 / 共通ツールを導入する


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock96: prompt-based SST-2 sentiment analysis")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名 / モデル名
    parser.add_argument("--dev-path", default=str(DEFAULT_DEV_PATH), help="SST-2 dev.tsv path")  # 开发数据路径 / 開発データパス
    parser.add_argument("--max-examples", type=int, default=None, help="limit examples for quick check")  # 样本上限 / サンプル上限
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    device = get_device()  # 获取设备 / デバイスを取得する
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読む
    model = load_causal_lm(args.model_name, device=device).eval()  # 读取模型 / モデルを読む
    rows = read_sst2_rows(args.dev_path, args.max_examples)  # 读取开发集 / 開発セットを読む
    correct = 0  # 正确数 / 正解数
    for row in tqdm(rows, desc="predict"):  # 遍历样本 / サンプルを走査する
        pred, neg_score, pos_score = prompt_sentiment_predict(tokenizer, model, row["text"], device=device)  # 预测情感 / 感情を予測する
        correct += int(pred == row["label"])  # 累加正确数 / 正解数を加算する
    accuracy = correct / max(len(rows), 1)  # 计算正解率 / 正解率を計算する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 96: Prompt-based Sentiment")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(f"examples: {len(rows)}")  # 输出样本数 / サンプル数を出力する
    print(f"accuracy: {accuracy:.6f}")  # 输出正解率 / 正解率を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ



# device: cuda; prompt-based zero/few-shot sentiment classification on 200 SST-2 examples.
# examples: 200
# accuracy: 0.735000
# Result is usable as a baseline but weaker than the fine-tuned classifier in knock97.
