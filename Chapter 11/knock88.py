'''
knock88.py: 極性分析 / 極性分析

問題87で保存したモデルを用いて、指定文の極性を予測する。
/ 問題87で保存したモデルを用いて、指定文の極性を予測する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ
from pathlib import Path  # 路径处理类 / パス処理クラス

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # 导入Transformers类 / Transformersクラスを導入する

from chapter11_utils import DEFAULT_CHECKPOINT_DIR, DEFAULT_SENTENCES, configure_quiet_mode, get_device  # 导入共用工具 / 共通ツールを導入する


def load_model_and_tokenizer(model_dir, device):  # 读取问题87保存的微调模型 / 問題87で保存した微調整モデルを読む
    model_dir = Path(model_dir)  # 转换为Path / Pathへ変換する
    if not model_dir.exists() or not (model_dir / "config.json").exists():  # 如果保存目录不存在 / 保存ディレクトリが存在しない場合
        raise FileNotFoundError(f"Fine-tuned checkpoint was not found: {model_dir}")  # 要求先运行87 / 先に87を実行するよう明確にする
    tokenizer = AutoTokenizer.from_pretrained(model_dir)  # 从保存目录读取tokenizer / 保存先からtokenizerを読む
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)  # 读取微调模型 / 微調整モデルを読む
    return model, tokenizer  # 返回模型和tokenizer / モデルとtokenizerを返す


def predict_polarity(model, tokenizer, sentence, device, max_length=128):
    model.eval()
    encoding = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length)
    encoding = {key: value.to(device) for key, value in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    label_id = int(probabilities.argmax().item())
    label_name = "positive" if label_id == 1 else "negative"
    return label_name


def main():  # 定义主函数 / メイン関数を定義する
    configure_quiet_mode()  # 抑制额外日志 / 余計なログを抑制する
    parser = argparse.ArgumentParser(description="knock88: predict polarity with fine-tuned BERT")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-dir", default=str(DEFAULT_CHECKPOINT_DIR), help="fine-tuned model directory")  # 模型目录 / モデルディレクトリ
    parser.add_argument("--sentence", action="append", dest="sentences", help="sentence to classify (repeatable)")  # 输入句子 / 入力文
    args = parser.parse_args()  # 解析命令行参数 / コマンドライン引数を解析する

    device = get_device()  # 获取设备 / デバイスを取得する
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)  # 读取模型和tokenizer / モデルとtokenizerを読む
    sentences = args.sentences if args.sentences else DEFAULT_SENTENCES  # 使用题目给定句子 / 課題指定の文を使う
    for sentence in sentences:  # 逐句预测 / 文ごとに予測する
        label = predict_polarity(model, tokenizer, sentence, device=device)  # 预测极性 / 極性を予測する
        print(f"{sentence}\t{label}")  # 输出句子和标签 / 文とラベルを出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
Execution result:
The movie was full of incomprehensibilities.	negative
The movie was full of fun.	positive
The movie was full of excitement.	positive
The movie was full of crap.	negative
The movie was full of rubbish.	negative
'''
