'''
knock92.py: 機械翻訳モデルの適用 / 機械翻訳モデルの適用

91で学習したモデルを用いて、任意の日本語文を英語に翻訳する。
/ 91で学習したモデルを用いて、任意の日本語文を英語に翻訳する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # PyTorch / PyTorch

from chapter13_utils import CHECKPOINT_DIR, get_device, get_ja_tagger, greedy_decode, load_checkpoint, tokenize_ja, encode  # 共用工具 / 共通ツール


def main():  # 主函数 / メイン関数
    parser = argparse.ArgumentParser(description="knock92: translate Japanese sentence")  # 参数解析 / 引数解析
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "transformer_mt.pt"), help="model checkpoint")  # checkpoint路径 / checkpointパス
    parser.add_argument("--sentence", default="猫 が 好き です 。", help="Japanese sentence")  # 输入日语 / 入力日本語
    parser.add_argument("--max-len", type=int, default=80, help="maximum output length")  # 最大长度 / 最大長
    args = parser.parse_args()  # 解析 / 解析する
    device = get_device()  # 设备 / デバイス
    model, src_vocab, tgt_vocab, _config = load_checkpoint(args.checkpoint, device)  # 读取模型 / モデルを読む
    tokens = tokenize_ja(args.sentence, get_ja_tagger())  # 日语分词 / 日本語分かち書き
    src_ids = torch.tensor(encode(tokens, src_vocab), dtype=torch.long).unsqueeze(1)  # 编码输入 / 入力を符号化
    output_tokens = greedy_decode(model, src_ids, src_vocab, tgt_vocab, args.max_len, device)  # 翻译 / 翻訳する
    print("=" * 50)  # 分隔线 / 区切り線
    print("Knock 92: Translation")  # 标题 / タイトル
    print("=" * 50)  # 分隔线 / 区切り線
    print(f"source tokens: {tokens}")  # 输出源token / 入力tokenを出力
    print(" ".join(output_tokens))  # 输出翻译 / 翻訳を出力


if __name__ == "__main__":  # 直接运行 / 直接実行
    main()  # 调用主函数 / メイン関数を呼ぶ


# AIX実行結果メモ (2026-07-18, log: ~/100knock/logs/ch13_knock92.log)
# checkpoint used: Chapter 13/checkpoints/transformer_mt.pt.
# source tokens: ['京都', 'は', '美しい', '。']
# output: his secular name was <unk>
# Observation: translation quality is poor for this example, consistent with the low BLEU in knock93.
