'''
knock80.py: トークン化 / トークン化

"The movie was full of incomprehensibilities."をBERT tokenizerで分解する。
/ "The movie was full of incomprehensibilities."をBERT tokenizerで分解する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from transformers import AutoTokenizer  # 导入tokenizer自动加载类 / tokenizer自動読み込みクラスを導入する


MODEL_NAME = "google-bert/bert-base-uncased"  # 指定使用的BERT模型 / 使用するBERTモデルを指定する
TOKENIZE_SENTENCE = "The movie was full of incomprehensibilities."  # 问题80的句子 / 問題80の文


def tokenize_text(tokenizer, sentence=TOKENIZE_SENTENCE):  # 将句子分解为token / 文をtokenへ分解する
    return tokenizer.tokenize(sentence)  # 调用BERT tokenizer / BERT tokenizerを呼び出す


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock80: tokenize a sentence with BERT")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名称 / モデル名
    parser.add_argument("--sentence", default=TOKENIZE_SENTENCE, help="sentence to tokenize")  # 输入句子 / 入力文
    args = parser.parse_args()  # 解析命令行参数 / コマンドライン引数を解析する

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # 读取tokenizer / tokenizerを読み込む
    tokens = tokenize_text(tokenizer, args.sentence)  # 分解句子为token / 文をtokenへ分解する

    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 80: Tokenization")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(f"sentence: {args.sentence}")  # 输出原句 / 元の文を出力する
    print(f"tokens: {tokens}")  # 输出token列 / token列を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
実行結果 / 运行结果:
==================================================
Knock 80: Tokenization
==================================================
sentence: The movie was full of incomprehensibilities.
tokens: ['the', 'movie', 'was', 'full', 'of', 'inc', '##omp', '##re', '##hen', '##si', '##bilities', '.']
'''

