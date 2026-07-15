'''
knock93.py: BLEUスコアの計測 / BLEUスコアの計測

91で学習したモデルを用いて、評価データ上のBLEUスコアをsacreBLEUで測定する。
/ 91で学習したモデルを用いて、評価データ上のBLEUスコアをsacreBLEUで測定する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # PyTorch / PyTorch
from tqdm.auto import tqdm  # 进度条 / 進捗バー

from chapter13_utils import CHECKPOINT_DIR, OUTPUT_DIR, PROCESSED_DIR, corpus_bleu, encode, get_device, greedy_decode, load_checkpoint, load_token_lines  # 共用工具 / 共通ツール


def main():  # 主函数 / メイン関数
    parser = argparse.ArgumentParser(description="knock93: evaluate BLEU")  # 参数解析 / 引数解析
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "transformer_mt.pt"), help="model checkpoint")  # checkpoint路径 / checkpointパス
    parser.add_argument("--split", default="test", choices=["dev", "test"], help="evaluation split")  # 评价split / 評価split
    parser.add_argument("--max-examples", type=int, default=None, help="limit examples")  # 上限 / 上限
    args = parser.parse_args()  # 解析 / 解析する
    device = get_device()  # 设备 / デバイス
    model, src_vocab, tgt_vocab, _config = load_checkpoint(args.checkpoint, device)  # 读取模型 / モデルを読む
    src_lines = load_token_lines(PROCESSED_DIR / f"{args.split}.ja.tok")[: args.max_examples]  # 读取源 / 入力を読む
    ref_lines = [" ".join(line) for line in load_token_lines(PROCESSED_DIR / f"{args.split}.en.tok")[: args.max_examples]]  # 读取参考 / 参照訳を読む
    predictions = []  # 初始化预测列表 / 予測リストを初期化する
    for tokens in tqdm(src_lines, desc="translate"):  # 遍历句子 / 文を走査する
        src_ids = torch.tensor(encode(tokens, src_vocab), dtype=torch.long).unsqueeze(1)  # 编码源 / 入力を符号化
        out_tokens = greedy_decode(model, src_ids, src_vocab, tgt_vocab, device=device)  # 翻译 / 翻訳
        predictions.append(" ".join(out_tokens))  # 保存预测 / 予測を保存
    score = corpus_bleu(predictions, ref_lines)  # 计算BLEU / BLEUを計算
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 创建输出目录 / 出力ディレクトリ作成
    (OUTPUT_DIR / f"{args.split}_predictions.txt").write_text("\n".join(predictions), encoding="utf-8")  # 保存预测 / 予測を保存
    print("=" * 50)  # 分隔线 / 区切り線
    print("Knock 93: BLEU")  # 标题 / タイトル
    print("=" * 50)  # 分隔线 / 区切り線
    print(f"BLEU: {score:.2f}")  # 输出BLEU / BLEUを出力


if __name__ == "__main__":  # 直接运行 / 直接実行
    main()  # 调用主函数 / メイン関数を呼ぶ

