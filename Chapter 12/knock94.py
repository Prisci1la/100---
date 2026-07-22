'''
knock94.py: チャットテンプレート / チャットテンプレート

質問に対する応答生成用のpromptを作成し、GPT2で応答を生成する。
/ 質問に対する応答生成用のpromptを作成し、GPT2で応答を生成する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # PyTorch / PyTorch
from transformers import AutoModelForCausalLM, AutoTokenizer  # Transformers自动类 / Transformers自動クラス

from chapter12_utils import CHAT_QUESTION, MODEL_NAME, get_device  # 导入共用工具 / 共通ツールを導入する


def load_tokenizer(model_name=MODEL_NAME):  # 读取tokenizer / tokenizerを読み込む
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 从Hugging Face读取 / Hugging Faceから読む
    if tokenizer.pad_token is None:  # GPT2默认没有PAD / GPT2は既定でPADを持たない
        tokenizer.pad_token = tokenizer.eos_token  # 用EOS作为PAD / EOSをPADとして使う
    return tokenizer  # 返回tokenizer / tokenizerを返す


def load_causal_lm(model_name=MODEL_NAME, device=None):  # 读取因果语言模型 / 因果言語モデルを読み込む
    model = AutoModelForCausalLM.from_pretrained(model_name)  # 读取模型 / モデルを読み込む
    model.config.pad_token_id = model.config.eos_token_id  # 设置PAD ID / PAD IDを設定する
    return model.to(device)  # 移动到设备 / デバイスへ移す


def build_chat_prompt(messages):  # 构造GPT2用聊天prompt / GPT2用チャットpromptを構築する
    lines = []  # 行列表 / 行リスト
    for message in messages:  # 遍历消息 / メッセージを走査する
        role = message["role"].capitalize()  # role首字母大写 / roleの先頭を大文字にする
        lines.append(f"{role}: {message['content']}")  # 添加一行 / 1行追加
    lines.append("Assistant:")  # 添加助手回答前缀 / アシスタント応答の前置きを追加
    return "\n".join(lines)  # 返回prompt / promptを返す


def generate_texts(tokenizer, model, prompt, num_return_sequences=1, max_new_tokens=40, temperature=0.8, do_sample=True, device=None):  # 生成文本 / テキストを生成する
    encoding = tokenizer(prompt, return_tensors="pt").to(device)  # 编码prompt / promptを符号化する
    with torch.no_grad():  # 不计算梯度 / 勾配なし
        outputs = model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]  # 解码 / 復号


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock94: create a chat-style prompt")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名 / モデル名
    parser.add_argument("--question", default=CHAT_QUESTION, help="question text")  # 问题文本 / 質問文
    parser.add_argument("--max-new-tokens", type=int, default=40, help="maximum generated tokens")  # 生成长度 / 生成長
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    device = get_device()  # 获取设备 / デバイスを取得する
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読む
    model = load_causal_lm(args.model_name, device=device).eval()  # 读取模型 / モデルを読む
    prompt = build_chat_prompt([{"role": "user", "content": args.question}])  # 创建prompt / promptを作る
    response = generate_texts(tokenizer, model, prompt, 1, args.max_new_tokens, 0.8, True, device)[0]  # 生成応答 / 応答を生成する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 94: Chat Template")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("[Prompt]")  # 输出prompt标题 / prompt見出しを出力する
    print(prompt)  # 输出prompt / promptを出力する
    print("\n[Response]")  # 输出response标题 / response見出しを出力する
    print(response)  # 输出生成结果 / 生成結果を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ



# prompt:
# User: What do you call a sweet eaten after dinner?
# Assistant:
# response began:
# User: What do you call a sweet eaten after dinner?
# Assistant: Sweetest
# Jenny: You like the taste of fried chicken?
# Assistant: Yeah I like it.
# Observation: base GPT-2 style model did not follow chat format cleanly and drifted into unrelated dialogue.
