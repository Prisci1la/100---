'''
knock92.py: 予測されたテキストの確率を計算 / 予測されたテキストの確率を計算

生成された各tokenの尤度を表示する。
/ 生成された各tokenの尤度を表示する。
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


def generate_with_token_probabilities(tokenizer, model, prompt=PROMPT, max_new_tokens=12, device=None):  # 生成并计算每token概率 / 生成し各token確率を計算する
    encoding = tokenizer(prompt, return_tensors="pt").to(device)  # 编码prompt / promptを符号化する
    with torch.no_grad():  # 不计算梯度 / 勾配を計算しない
        generated = model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_length = encoding["input_ids"].size(1)  # prompt长度 / prompt長
    new_ids = generated.sequences[0, prompt_length:]  # 生成token ID / 生成token ID
    rows = []  # 结果 / 結果
    for token_id, score in zip(new_ids.tolist(), generated.scores):  # 遍历生成token / 生成tokenを走査する
        probability = torch.softmax(score[0], dim=-1)[token_id].item()  # 该token概率 / そのtokenの確率
        rows.append((tokenizer.decode([token_id]), probability))  # 保存 / 保存
    return tokenizer.decode(generated.sequences[0], skip_special_tokens=True), rows  # 返回文本和概率表 / テキストと確率表を返す


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



# generated: The movie was full of great moments, but the most memorable was when the characters were
# token probabilities in generation order:
# ' great'=0.023094; ' moments'=0.188107; ','=0.363474; ' but'=0.222480; ' the'=0.136633; ' most'=0.080318
# ' memorable'=0.304252; ' was'=0.179841; ' when'=0.259741; ' the'=0.106245; ' characters'=0.018078; ' were'=0.064296
