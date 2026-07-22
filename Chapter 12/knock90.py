'''
knock90.py: 次単語予測 / 次単語予測

"The movie was full of"に続くtoken上位10個と確率を求める。
/ "The movie was full of"に続くtoken上位10個と確率を求める。
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


def next_token_topk(tokenizer, model, prompt=PROMPT, top_k=10, device=None):  # 预测下一个token top-k / 次token top-kを予測する
    encoding = tokenizer(prompt, return_tensors="pt").to(device)  # 编码prompt / promptを符号化する
    with torch.no_grad():  # 不计算梯度 / 勾配を計算しない
        logits = model(**encoding).logits[0, -1]  # 取最后位置logit / 最後位置のlogitを取る
        probabilities = torch.softmax(logits, dim=-1)  # 转换为概率 / 確率へ変換する
        top_probs, top_ids = torch.topk(probabilities, k=top_k)  # 取top-k / top-kを取る
    tokens = [tokenizer.decode([token_id]) for token_id in encoding["input_ids"][0].tolist()]  # prompt token列 / prompt token列
    predictions = [(tokenizer.decode([idx]).strip(), prob.item()) for idx, prob in zip(top_ids.tolist(), top_probs)]  # 文本化 / テキスト化
    return tokens, predictions  # 返回prompt token和预测 / prompt tokenと予測を返す


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
