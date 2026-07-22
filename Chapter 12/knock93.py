'''
knock93.py: パープレキシティ / パープレキシティ

複数の文に対して、事前学習済みGPT2のperplexityを計測する。
/ 複数の文に対して、事前学習済みGPT2のperplexityを計測する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ
import math  # 数学函数 / 数学関数

import torch  # PyTorch / PyTorch
from transformers import AutoModelForCausalLM, AutoTokenizer  # Transformers自动类 / Transformers自動クラス

from chapter12_utils import MODEL_NAME, PPL_SENTENCES, get_device  # 导入共用工具 / 共通ツールを導入する


def load_tokenizer(model_name=MODEL_NAME):  # 读取tokenizer / tokenizerを読み込む
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 从Hugging Face读取 / Hugging Faceから読む
    if tokenizer.pad_token is None:  # GPT2默认没有PAD / GPT2は既定でPADを持たない
        tokenizer.pad_token = tokenizer.eos_token  # 用EOS作为PAD / EOSをPADとして使う
    return tokenizer  # 返回tokenizer / tokenizerを返す


def load_causal_lm(model_name=MODEL_NAME, device=None):  # 读取因果语言模型 / 因果言語モデルを読み込む
    model = AutoModelForCausalLM.from_pretrained(model_name)  # 读取模型 / モデルを読み込む
    model.config.pad_token_id = model.config.eos_token_id  # 设置PAD ID / PAD IDを設定する
    return model.to(device)  # 移动到设备 / デバイスへ移す


def sentence_perplexity(tokenizer, model, sentence, device=None):  # 计算句子perplexity / 文のperplexityを計算する
    encoding = tokenizer(sentence, return_tensors="pt").to(device)  # 编码句子 / 文を符号化する
    with torch.no_grad():  # 不计算梯度 / 勾配を計算しない
        loss = model(**encoding, labels=encoding["input_ids"]).loss  # 语言模型loss / 言語モデルloss
    return math.exp(loss.item())  # perplexity = exp(loss)


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
