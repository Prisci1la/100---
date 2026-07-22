'''
knock96.py: プロンプトによる感情分析 / プロンプトによる感情分析

ファインチューニングなしで、promptによるSST-2開発データの正解率を計測する。
/ ファインチューニングなしで、promptによるSST-2開発データの正解率を計測する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # PyTorch / PyTorch
from tqdm.auto import tqdm  # 导入进度条 / 進捗バーを導入する
from transformers import AutoModelForCausalLM, AutoTokenizer  # Transformers自动类 / Transformers自動クラス

from chapter12_utils import DEFAULT_DEV_PATH, MODEL_NAME, get_device, read_sst2_rows  # 导入共用工具 / 共通ツールを導入する


def load_tokenizer(model_name=MODEL_NAME):  # 读取tokenizer / tokenizerを読み込む
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 从Hugging Face读取 / Hugging Faceから読む
    if tokenizer.pad_token is None:  # GPT2默认没有PAD / GPT2は既定でPADを持たない
        tokenizer.pad_token = tokenizer.eos_token  # 用EOS作为PAD / EOSをPADとして使う
    return tokenizer  # 返回tokenizer / tokenizerを返す


def load_causal_lm(model_name=MODEL_NAME, device=None):  # 读取因果语言模型 / 因果言語モデルを読み込む
    model = AutoModelForCausalLM.from_pretrained(model_name)  # 读取模型 / モデルを読み込む
    model.config.pad_token_id = model.config.eos_token_id  # 设置PAD ID / PAD IDを設定する
    return model.to(device)  # 移动到设备 / デバイスへ移す


def sentiment_prompt(text):  # 构造情感分析prompt / 感情分析promptを作る
    return f"Review: {text}\nSentiment:"  # 返回prompt / promptを返す


def continuation_log_probability(tokenizer, model, prompt, continuation, device=None):  # 计算续写log概率 / 続きのlog確率を計算する
    full_text = prompt + continuation  # prompt和候选标签合并 / promptと候補ラベルを結合
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)  # 编码全文 / 全文を符号化
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.size(1)  # prompt token数 / prompt token数
    with torch.no_grad():  # 不计算梯度 / 勾配なし
        logits = model(full_ids).logits[:, :-1, :]  # 下一个token预测logits / 次token予測logits
        target = full_ids[:, 1:]  # 目标token / 目標token
        log_probs = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)  # 目标token log概率 / 目標token log確率
    start = max(prompt_len - 1, 0)  # continuation起点 / continuation開始位置
    return log_probs[0, start:].sum().item()  # 返回续写log概率总和 / 続きのlog確率合計


def prompt_sentiment_predict(tokenizer, model, text, device=None):  # 用prompt预测情感 / promptで感情を予測する
    prompt = sentiment_prompt(text)  # 构造prompt / promptを作る
    neg_score = continuation_log_probability(tokenizer, model, prompt, " negative", device=device)  # negative得分 / negativeスコア
    pos_score = continuation_log_probability(tokenizer, model, prompt, " positive", device=device)  # positive得分 / positiveスコア
    return 1 if pos_score > neg_score else 0, neg_score, pos_score  # 返回预测和分数 / 予測とスコアを返す


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock96: prompt-based SST-2 sentiment analysis")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名 / モデル名
    parser.add_argument("--dev-path", default=str(DEFAULT_DEV_PATH), help="SST-2 dev.tsv path")  # 开发数据路径 / 開発データパス
    parser.add_argument("--max-examples", type=int, default=None, help="limit examples for quick check")  # 样本上限 / サンプル上限
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    device = get_device()  # 获取设备 / デバイスを取得する
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読む
    model = load_causal_lm(args.model_name, device=device).eval()  # 读取模型 / モデルを読む
    rows = read_sst2_rows(args.dev_path, args.max_examples)  # 读取开发集 / 開発セットを読む
    correct = 0  # 正确数 / 正解数
    for row in tqdm(rows, desc="predict"):  # 遍历样本 / サンプルを走査する
        pred, neg_score, pos_score = prompt_sentiment_predict(tokenizer, model, row["text"], device=device)  # 预测情感 / 感情を予測する
        correct += int(pred == row["label"])  # 累加正确数 / 正解数を加算する
    accuracy = correct / max(len(rows), 1)  # 计算正解率 / 正解率を計算する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 96: Prompt-based Sentiment")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(f"examples: {len(rows)}")  # 输出样本数 / サンプル数を出力する
    print(f"accuracy: {accuracy:.6f}")  # 输出正解率 / 正解率を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ



# device: cuda; prompt-based zero/few-shot sentiment classification on 200 SST-2 examples.
# examples: 200
# accuracy: 0.735000
# Result is usable as a baseline but weaker than the fine-tuned classifier in knock97.
