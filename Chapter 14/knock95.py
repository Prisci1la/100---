'''
knock95.py: サブワード化 / サブワード化

SentencePieceで日英データをサブワード化し、91-94を再実行するためのデータを作成する。
/ SentencePieceで日英データをサブワード化し、91-94を再実行するためのデータを作成する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import matplotlib.pyplot as plt  # 绘图库 / 描画ライブラリ
import torch  # PyTorch / PyTorch
from torch import nn  # 神经网络模块 / ニューラルネットワーク
from torch.utils.data import DataLoader  # DataLoader / DataLoader
from torch.utils.data.distributed import DistributedSampler  # DDP sampler / DDP sampler
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP / DDP

from chapter14_utils import CHECKPOINT_DIR, CH13_PROCESSED_DIR, DATA_DIR, OUTPUT_DIR, TransformerMT, TranslationDataset, build_vocab, cleanup_distributed, collate_translation, encode_with_sentencepiece, evaluate_bleu_for_beam, get_device, is_main_process, load_token_lines, load_vocab, save_checkpoint, save_vocab, setup_distributed, train_epoch, train_sentencepiece, write_json  # 共用工具 / 共通ツール


def main():  # 主函数 / メイン関数
    parser = argparse.ArgumentParser(description="knock95: train SentencePiece and encode KFTT")  # 参数解析 / 引数解析
    parser.add_argument("--vocab-size", type=int, default=8000, help="SentencePiece vocabulary size")  # SP词表大小 / SP語彙サイズ
    parser.add_argument("--min-freq", type=int, default=1, help="vocab minimum frequency")  # 最低频率 / 最低頻度
    parser.add_argument("--epochs", type=int, default=5, help="subword Transformer training epochs")  # epoch数 / epoch数
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")  # batch大小 / batchサイズ
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")  # 学习率 / 学習率
    parser.add_argument("--beam-sizes", default="1,2,5,10,20,50,100", help="comma separated beam sizes")  # beam候选 / beam候補
    parser.add_argument("--max-train-examples", type=int, default=None, help="limit train examples for quick check")  # 训练上限 / 訓練上限
    parser.add_argument("--max-dev-examples", type=int, default=None, help="limit dev examples for quick check")  # 开发上限 / 開発上限
    parser.add_argument("--prepare-only", action="store_true", help="only create SentencePiece data")  # 只准备数据 / データ作成のみ
    args = parser.parse_args()  # 解析 / 解析する
    distributed, rank, local_rank, world_size = setup_distributed()  # DDP初期化 / DDP初始化
    sp_dir = DATA_DIR / "sentencepiece"  # SP目录 / SPディレクトリ
    out_dir = DATA_DIR / "subword"  # 输出目录 / 出力ディレクトリ
    try:
        if is_main_process():
            ja_model = train_sentencepiece(CH13_PROCESSED_DIR / "train.ja.tok", sp_dir / "kftt_ja", args.vocab_size)  # 训练日语SP / 日本語SPを学習する
            en_model = train_sentencepiece(CH13_PROCESSED_DIR / "train.en.tok", sp_dir / "kftt_en", args.vocab_size)  # 训练英语SP / 英語SPを学習する
            for split in ["train", "dev", "test"]:  # 编码各split / 各splitを符号化する
                encode_with_sentencepiece(ja_model, CH13_PROCESSED_DIR / f"{split}.ja.tok", out_dir / f"{split}.ja.sp")  # 日语SP编码 / 日本語SP符号化
                encode_with_sentencepiece(en_model, CH13_PROCESSED_DIR / f"{split}.en.tok", out_dir / f"{split}.en.sp")  # 英语SP编码 / 英語SP符号化
            src_vocab = build_vocab(load_token_lines(out_dir / "train.ja.sp"), args.vocab_size, args.min_freq)  # 构建源词表 / 入力語彙を作る
            tgt_vocab = build_vocab(load_token_lines(out_dir / "train.en.sp"), args.vocab_size, args.min_freq)  # 构建目标词表 / 目標語彙を作る
            save_vocab(src_vocab, out_dir / "vocab.ja.json")  # 保存源词表 / 入力語彙を保存する
            save_vocab(tgt_vocab, out_dir / "vocab.en.json")  # 保存目标词表 / 目標語彙を保存する
            print("=" * 50)  # 分隔线 / 区切り線
            print("Knock 95: SentencePiece")  # 标题 / タイトル
            print("=" * 50)  # 分隔线 / 区切り線
            print(f"ja model: {ja_model}")  # 输出模型路径 / モデルパスを出力
            print(f"en model: {en_model}")  # 输出模型路径 / モデルパスを出力
            print(f"subword data: {out_dir}")  # 输出数据路径 / データパスを出力
        if distributed:
            torch.distributed.barrier()
        if args.prepare_only:  # 只准备数据 / データ作成のみ
            return  # 结束 / 終了

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu") if distributed else get_device()  # 设备 / デバイス
        if device.type != "cuda":  # 题目要求GPU训练 / 課題はGPU学習を要求する
            raise RuntimeError("knock95 requires GPU training. CUDA is not available.")  # 报错 / エラー
        src_vocab = load_vocab(out_dir / "vocab.ja.json")  # 读取源词表 / 入力語彙を読む
        tgt_vocab = load_vocab(out_dir / "vocab.en.json")  # 读取目标词表 / 目標語彙を読む
        train_src = load_token_lines(out_dir / "train.ja.sp")[: args.max_train_examples]  # 训练源 / 訓練入力
        train_tgt = load_token_lines(out_dir / "train.en.sp")[: args.max_train_examples]  # 训练目标 / 訓練目標
        dev_src = load_token_lines(out_dir / "dev.ja.sp")  # 开发源 / 開発入力
        dev_ref = [" ".join(line) for line in load_token_lines(out_dir / "dev.en.sp")]  # 开发参考 / 開発参照
        dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, collate_fn=collate_translation)  # loader / loader
        config = {"emb_size": 256, "nhead": 4, "num_layers": 3, "dim_feedforward": 512}  # 模型配置 / モデル設定
        model = TransformerMT(len(src_vocab), len(tgt_vocab), **config).to(device)  # 创建模型 / モデルを作る
        if distributed:
            model = DDP(model, device_ids=[local_rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 优化器 / 最適化器
        loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])  # loss函数 / loss関数
        for epoch in range(1, args.epochs + 1):  # 训练循环 / 学習ループ
            if sampler is not None:
                sampler.set_epoch(epoch)
            loss = train_epoch(model, loader, optimizer, loss_fn, device)  # 训练一轮 / 1epoch学習
            if is_main_process():
                print(f"subword epoch {epoch:02d}: loss={loss:.6f}")  # 输出loss / lossを出力
        if is_main_process():
            save_checkpoint(CHECKPOINT_DIR / "knock95_subword_mt.pt", model, src_vocab, tgt_vocab, config)  # 保存模型 / モデルを保存

        if is_main_process():
            beam_sizes = [int(value) for value in args.beam_sizes.split(",")]  # beam列表 / beam一覧
            if any(beam < 1 for beam in beam_sizes):  # 检查beam / beam確認
                raise ValueError("beam sizes must be positive integers")  # 报错 / エラー
            eval_model = model.module if hasattr(model, "module") else model
            scores = []  # BLEU结果 / BLEU結果
            for beam in beam_sizes:  # 遍历beam / beamを走査
                score, _predictions = evaluate_bleu_for_beam(eval_model, dev_src, dev_ref, src_vocab, tgt_vocab, beam, args.max_dev_examples, device)  # 评价BLEU / BLEU評価
                scores.append(score)  # 保存分数 / スコア保存
                print(f"subword beam={beam}: BLEU={score:.2f}")  # 输出BLEU / BLEUを出力
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 输出目录 / 出力ディレクトリ
            write_json(OUTPUT_DIR / "knock95_subword_beam_bleu.json", dict(zip(beam_sizes, scores)))  # 保存JSON / JSON保存
            plt.figure(figsize=(8, 5))  # 创建图 / 図を作る
            plt.plot(beam_sizes, scores, marker="o")  # 绘制曲线 / 曲線を描く
            plt.xlabel("Beam size")  # X标签 / Xラベル
            plt.ylabel("BLEU")  # Y标签 / Yラベル
            plt.tight_layout()  # 调整布局 / レイアウト調整
            plt.savefig(OUTPUT_DIR / "knock95_subword_beam_bleu.png", dpi=120)  # 保存图 / 図を保存
    finally:
        cleanup_distributed()


if __name__ == "__main__":  # 直接运行 / 直接実行
    main()  # 调用主函数 / メイン関数を呼ぶ


# AIX実行結果メモ (2026-07-19, logs: ch14_knock95*_g24.log)
# original bs16 run was stopped during beam=10 evaluation; limited g24 eval completed.
# limited eval subset size was 200; sacreBLEU warned that tokenized periods may indicate non-detokenized test data.
# subword beam BLEU:
# beam=1 -> 6.47, completed 200/200 in about 00:28
# beam=2 -> 7.87, completed 200/200 in about 01:00
# beam=5 -> 8.04
# beam=10 -> 8.48
# outputs: Chapter 14/outputs/knock95_subword_beam_bleu_limited.json and .png.
