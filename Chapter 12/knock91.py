'''
knock91.py: 続きのテキストの予測 / 続きのテキストの予測

デコーディング方法やtemperatureを変えながら、複数の続きを生成する。
/ デコーディング方法やtemperatureを変えながら、複数の続きを生成する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # PyTorch / PyTorch
from transformers import AutoModelForCausalLM, AutoTokenizer  # Transformers自动类 / Transformers自動クラス

from chapter12_utils import MODEL_NAME, PROMPT, get_device  # 导入共用工具 / 共通ツールを導入する


def load_tokenizer(model_name=MODEL_NAME):  # 读取tokenizer / tokenizerを読み込む
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 从Hugging Face读取 / Hugging Faceから読む
    if tokenizer.pad_token is None:  # GPT2默认没有PAD / GPT2は既定でPADを持たない
        tokenizer.pad_token = tokenizer.eos_token  # 用EOS作为PAD / EOSをPADとして使う
    return tokenizer  # 返回tokenizer / tokenizerを返す


def load_causal_lm(model_name=MODEL_NAME, device=None):  # 读取因果语言模型 / 因果言語モデルを読み込む
    model = AutoModelForCausalLM.from_pretrained(model_name)  # 读取模型 / モデルを読み込む
    model.config.pad_token_id = model.config.eos_token_id  # 设置PAD ID / PAD IDを設定する
    return model.to(device)  # 移动到设备 / デバイスへ移す


def generate_texts(tokenizer, model, prompt=PROMPT, num_return_sequences=5, max_new_tokens=20, temperature=1.0, do_sample=True, device=None):  # 生成多个续写 / 複数の続きを生成する
    encoding = tokenizer(prompt, return_tensors="pt").to(device)  # 编码prompt / promptを符号化する
    with torch.no_grad():  # 不计算梯度 / 勾配を計算しない
        outputs = model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]  # 解码生成结果 / 生成結果を復号する


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



# device: cuda; model: openai-community/gpt2-medium; prompt: "The movie was full of"
# temperature=0.7 outputs included:
# - "The movie was full of moments ... but they were so important to the story ..."
# - "The movie was full of them. But, most people were shocked to see this."
# - "The movie was full of references to the movie \"Superman Returns\" ..."
# temperature=1.0: "The movie was full of them, of the way the movie was made."
# temperature=1.0 also produced a positive review-like continuation about "great moments".
# temperature=1.3: "The movie was full of amazing action scenes ..." and more diverse but less stable continuations.
