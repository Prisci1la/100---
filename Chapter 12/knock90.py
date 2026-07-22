'''
knock90.py: 次単語予測 / 次単語予測

"The movie was full of"に続くtoken上位10個と確率を求める。
/ "The movie was full of"に続くtoken上位10個と確率を求める。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from chapter12_utils import MODEL_NAME, PROMPT, get_device, load_causal_lm, load_tokenizer, next_token_topk  # 导入共用工具 / 共通ツールを導入する


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock90: next token prediction")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名 / モデル名
    parser.add_argument("--prompt", default=PROMPT, help="prompt text")  # prompt文本 / promptテキスト
    parser.add_argument("--top-k", type=int, default=10, help="number of tokens")  # top-k数量 / top-k数
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    device = get_device()  # 获取设备 / デバイスを取得する
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読む
    model = load_causal_lm(args.model_name, device=device).eval()  # 读取模型 / モデルを読む
    prompt_tokens, predictions = next_token_topk(tokenizer, model, args.prompt, args.top_k, device=device)  # 预测下一个token / 次tokenを予測する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 90: Next Token Prediction")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(f"device: {device}")  # 输出设备 / デバイスを出力する
    print(f"prompt tokens: {prompt_tokens}")  # 输出prompt token列 / prompt token列を出力する
    for rank, (token, probability) in enumerate(predictions, start=1):  # 遍历结果 / 結果を走査する
        print(f"{rank:02d}\t{token}\t{probability:.6f}")  # 输出排名 / 順位を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ



# device: cuda; visible GPUs on g20 were 8 x NVIDIA TITAN RTX (23.64GiB each).
# prompt tokens: ['The', ' movie', ' was', ' full', ' of']
# top 10 next-token probabilities:
# 01 great=0.023094; 02 references=0.013512; 03 action=0.013043; 04 moments=0.012450; 05 the=0.011860
# 06 characters=0.008720; 07 these=0.007216; 08 surprises=0.006894; 09 fun=0.006526; 10 them=0.006154
