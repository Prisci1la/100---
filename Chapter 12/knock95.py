'''
knock95.py: マルチターンのチャット / マルチターンのチャット

問題94の応答に続けて追加質問を行い、その時のpromptと応答を表示する。
/ 問題94の応答に続けて追加質問を行い、その時のpromptと応答を表示する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from chapter12_utils import CHAT_QUESTION, FOLLOWUP_QUESTION, MODEL_NAME, build_chat_prompt, generate_texts, get_device, load_causal_lm, load_tokenizer  # 导入共用工具 / 共通ツールを導入する


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock95: multi-turn chat generation")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名 / モデル名
    parser.add_argument("--question", default=CHAT_QUESTION, help="first question")  # 第一问题 / 最初の質問
    parser.add_argument("--followup", default=FOLLOWUP_QUESTION, help="follow-up question")  # 追加问题 / 追加質問
    parser.add_argument("--max-new-tokens", type=int, default=40, help="maximum generated tokens")  # 生成长度 / 生成長
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    device = get_device()  # 获取设备 / デバイスを取得する
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読む
    model = load_causal_lm(args.model_name, device=device).eval()  # 读取模型 / モデルを読む
    first_prompt = build_chat_prompt([{"role": "user", "content": args.question}])  # 创建第一轮prompt / 1回目promptを作る
    first_answer = generate_texts(tokenizer, model, first_prompt, 1, args.max_new_tokens, 0.8, True, device)[0].split("Assistant:", 1)[-1].strip()  # 生成第一轮回答 / 1回目の回答を生成する
    messages = [  # 构造多轮消息 / 複数ターンメッセージを作る
        {"role": "user", "content": args.question},
        {"role": "assistant", "content": first_answer},
        {"role": "user", "content": args.followup},
    ]
    second_prompt = build_chat_prompt(messages)  # 创建第二轮prompt / 2回目promptを作る
    second_answer = generate_texts(tokenizer, model, second_prompt, 1, args.max_new_tokens, 0.8, True, device)[0]  # 生成第二轮回答 / 2回目の回答を生成する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 95: Multi-turn Chat")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("[Prompt]")  # 输出prompt标题 / prompt見出しを出力する
    print(second_prompt)  # 输出prompt / promptを出力する
    print("\n[Response]")  # 输出response标题 / response見出しを出力する
    print(second_answer)  # 输出生成结果 / 生成結果を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ

