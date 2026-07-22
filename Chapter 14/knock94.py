'''
knock94.py: ビーム探索 / ビーム探索

ビーム幅を変えながら開発セットBLEUを測定し、曲線をプロットする。
/ ビーム幅を変えながら開発セットBLEUを測定し、曲線をプロットする。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import matplotlib.pyplot as plt  # 绘图库 / 描画ライブラリ

from chapter14_utils import BASE_CHECKPOINT, CH13_PROCESSED_DIR, OUTPUT_DIR, evaluate_bleu_for_beam, get_device, load_checkpoint, load_token_lines, write_json  # 共用工具 / 共通ツール


def main():  # 主函数 / メイン関数
    parser = argparse.ArgumentParser(description="knock94: beam search BLEU curve")  # 参数解析 / 引数解析
    parser.add_argument("--checkpoint", default=str(BASE_CHECKPOINT), help="model checkpoint")  # checkpoint路径 / checkpointパス
    parser.add_argument("--beam-sizes", default="1,2,5,10,20,50,100", help="comma separated beam sizes")  # beam列表 / beam一覧
    parser.add_argument("--max-examples", type=int, default=None, help="limit dev examples for quick check")  # 样本上限 / サンプル上限
    args = parser.parse_args()  # 解析 / 解析する
    device = get_device()  # 设备 / デバイス
    model, src_vocab, tgt_vocab, _config = load_checkpoint(args.checkpoint, device)  # 读取模型 / モデルを読む
    src_lines = load_token_lines(CH13_PROCESSED_DIR / "dev.ja.tok")  # 读取开发源句 / 開発入力を読む
    ref_lines = [" ".join(line) for line in load_token_lines(CH13_PROCESSED_DIR / "dev.en.tok")]  # 读取参考译文 / 参照訳を読む
    beam_sizes = [int(value) for value in args.beam_sizes.split(",")]  # 解析beam / beamを解析する
    if any(beam < 1 for beam in beam_sizes):  # 检查beam / beamを確認する
        raise ValueError("beam sizes must be positive integers")  # 报错 / エラーにする
    scores = []  # 初始化分数 / スコアを初期化する
    for beam in beam_sizes:  # 遍历beam宽度 / beam幅を走査する
        score, _predictions = evaluate_bleu_for_beam(model, src_lines, ref_lines, src_vocab, tgt_vocab, beam, args.max_examples, device)  # 计算BLEU / BLEUを計算する
        scores.append(score)  # 保存分数 / スコアを保存する
        print(f"beam={beam}: BLEU={score:.2f}")  # 输出结果 / 結果を出力する
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 创建输出目录 / 出力ディレクトリ作成
    write_json(OUTPUT_DIR / "knock94_beam_bleu.json", dict(zip(beam_sizes, scores)))  # 保存JSON / JSONを保存する
    plt.figure(figsize=(8, 5))  # 创建图 / 図を作る
    plt.plot(beam_sizes, scores, marker="o")  # 绘制曲线 / 曲線を描く
    plt.xlabel("Beam size")  # X标签 / Xラベル
    plt.ylabel("BLEU")  # Y标签 / Yラベル
    plt.tight_layout()  # 调整布局 / レイアウト調整
    plt.savefig(OUTPUT_DIR / "knock94_beam_bleu.png", dpi=120)  # 保存图 / 図を保存する


if __name__ == "__main__":  # 直接运行 / 直接実行
    main()  # 调用主函数 / メイン関数を呼ぶ


# AIX実行結果メモ (2026-07-19, log: ~/100knock/logs/ch14_knock94_limited.log)
# limited evaluation used a reduced test subset to finish reliably.
# limited beam BLEU:
# beam=1 -> 1.6781618416538517
# beam=2 -> 2.56100191090642
# beam=5 -> 3.2662815026687344
# beam=10 -> 3.985241806807347
# outputs: Chapter 14/outputs/knock94_beam_bleu.json and knock94_beam_bleu.png.
