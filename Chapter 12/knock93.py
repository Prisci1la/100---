'''
knock93.py: パープレキシティ / パープレキシティ

複数の文に対して、事前学習済みGPT2のperplexityを計測する。
/ 複数の文に対して、事前学習済みGPT2のperplexityを計測する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from chapter12_utils import MODEL_NAME, PPL_SENTENCES, get_device, load_causal_lm, load_tokenizer, sentence_perplexity  # 导入共用工具 / 共通ツールを導入する


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock93: measure perplexity")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名 / モデル名
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    device = get_device()  # 获取设备 / デバイスを取得する
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読む
    model = load_causal_lm(args.model_name, device=device).eval()  # 读取模型 / モデルを読む
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 93: Perplexity")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    for sentence in PPL_SENTENCES:  # 遍历句子 / 文を走査する
        ppl = sentence_perplexity(tokenizer, model, sentence, device=device)  # 计算perplexity / perplexityを計算する
        print(f"{ppl:.4f}\t{sentence}")  # 输出结果 / 結果を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ



# device: cuda; model: openai-community/gpt2-medium.
# PPL: 89.4542 "The movie was full of surprises"
# PPL: 164.8894 "The movies were full of surprises"
# PPL: 324.1090 "The movie were full of surprises"
# PPL: 388.4485 "The movies was full of surprises"
# Lower perplexity matched the grammatical singular sentence.
