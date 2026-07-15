'''
knock91.py: 続きのテキストの予測 / 続きのテキストの予測

デコーディング方法やtemperatureを変えながら、複数の続きを生成する。
/ デコーディング方法やtemperatureを変えながら、複数の続きを生成する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from chapter12_utils import MODEL_NAME, PROMPT, get_device, generate_texts, load_causal_lm, load_tokenizer  # 导入共用工具 / 共通ツールを導入する


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock91: generate continuations")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名 / モデル名
    parser.add_argument("--prompt", default=PROMPT, help="prompt text")  # prompt文本 / promptテキスト
    parser.add_argument("--max-new-tokens", type=int, default=30, help="maximum generated tokens")  # 生成长度 / 生成長
    parser.add_argument("--num-return-sequences", type=int, default=3, help="number of samples")  # 生成数量 / 生成数
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    device = get_device()  # 获取设备 / デバイスを取得する
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読む
    model = load_causal_lm(args.model_name, device=device).eval()  # 读取模型 / モデルを読む
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 91: Text Generation")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    for temperature in [0.7, 1.0, 1.3]:  # 尝试不同temperature / 異なるtemperatureを試す
        print(f"\n[temperature={temperature}]")  # 输出条件 / 条件を出力する
        texts = generate_texts(tokenizer, model, args.prompt, args.num_return_sequences, args.max_new_tokens, temperature, True, device)  # 生成文本 / テキストを生成する
        for text in texts:  # 遍历生成结果 / 生成結果を走査する
            print(text)  # 输出文本 / テキストを出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ

