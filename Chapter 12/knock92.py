'''
knock92.py: 予測されたテキストの確率を計算 / 予測されたテキストの確率を計算

生成された各tokenの尤度を表示する。
/ 生成された各tokenの尤度を表示する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from chapter12_utils import MODEL_NAME, PROMPT, generate_with_token_probabilities, get_device, load_causal_lm, load_tokenizer  # 导入共用工具 / 共通ツールを導入する


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock92: generated token probabilities")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名 / モデル名
    parser.add_argument("--prompt", default=PROMPT, help="prompt text")  # prompt文本 / promptテキスト
    parser.add_argument("--max-new-tokens", type=int, default=12, help="maximum generated tokens")  # 生成长度 / 生成長
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    device = get_device()  # 获取设备 / デバイスを取得する
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読む
    model = load_causal_lm(args.model_name, device=device).eval()  # 读取模型 / モデルを読む
    text, rows = generate_with_token_probabilities(tokenizer, model, args.prompt, args.max_new_tokens, device)  # 生成并计算概率 / 生成して確率を計算する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 92: Generated Token Probabilities")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(text)  # 输出生成文本 / 生成文を出力する
    for token, probability in rows:  # 遍历token概率 / token確率を走査する
        print(f"{token!r}\t{probability:.6f}")  # 输出token和概率 / tokenと確率を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ

