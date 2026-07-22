'''
knock96.py: 学習過程の可視化 / 学習過程の可視化

TensorBoardで訓練lossと開発BLEUを記録しながら翻訳モデルを学習する。
/ TensorBoardで訓練lossと開発BLEUを記録しながら翻訳モデルを学習する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # PyTorch / PyTorch
import torch.distributed as dist  # 分布式 / 分散
from torch import nn  # 神经网络模块 / ニューラルネットワーク
from torch.utils.data import DataLoader  # DataLoader / DataLoader
from torch.utils.data.distributed import DistributedSampler  # DDP sampler / DDP sampler
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP / DDP
from torch.utils.tensorboard import SummaryWriter  # TensorBoard writer / TensorBoard writer

from chapter14_utils import CH13_PROCESSED_DIR, TransformerMT, TranslationDataset, cleanup_distributed, collate_translation, create_padding_mask, evaluate_bleu_for_beam, get_device, is_main_process, load_token_lines, load_vocab, setup_distributed, train_epoch  # 共用工具 / 共通ツール


def evaluate_loss(model, loader, loss_fn, device):  # 计算验证loss / 検証lossを計算する
    model.eval()  # 评价模式 / 評価モード
    total_loss = 0.0  # 累计loss / lossを累積
    with torch.no_grad():  # 不计算梯度 / 勾配なし
        for src, tgt in loader:  # 遍历batch / batchを走査
            src = src.to(device)  # 移动源句 / 入力を移動
            tgt = tgt.to(device)  # 移动目标句 / 目標を移動
            tgt_input = tgt[:-1, :]  # decoder输入 / decoder入力
            logits = model(src, tgt_input, create_padding_mask(src), create_padding_mask(tgt_input), create_padding_mask(src))  # 前向计算 / 順伝播
            target = tgt[1:, :].reshape(-1)  # 预测目标 / 予測対象
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target)  # loss / loss
            total_loss += loss.item()  # 累加 / 加算
    stats = torch.tensor([total_loss, len(loader)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1].clamp(min=1)).item()  # 平均loss / 平均loss


def main():  # 主函数 / メイン関数
    parser = argparse.ArgumentParser(description="knock96: TensorBoard visualization")  # 参数解析 / 引数解析
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")  # epoch数 / epoch数
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")  # batch大小 / batchサイズ
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")  # 学习率 / 学習率
    parser.add_argument("--log-dir", default="outputs/tensorboard", help="TensorBoard log directory")  # log目录 / logディレクトリ
    parser.add_argument("--max-train-examples", type=int, default=None, help="limit train examples")  # 训练上限 / 訓練上限
    parser.add_argument("--max-dev-examples", type=int, default=None, help="limit dev examples for quick BLEU check")  # 开发上限 / 開発上限
    parser.add_argument("--max-train-bleu-examples", type=int, default=None, help="limit train examples for BLEU logging")  # 训练BLEU上限 / 訓練BLEU上限
    args = parser.parse_args()  # 解析 / 解析する
    distributed, rank, local_rank, world_size = setup_distributed()
    try:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu") if distributed else get_device()  # 设备 / デバイス
        if device.type != "cuda":  # 题目要求GPU训练 / 課題はGPU学習を要求する
            raise RuntimeError("knock96 requires GPU training. CUDA is not available.")  # 报错 / エラー
        src_vocab = load_vocab(CH13_PROCESSED_DIR / "vocab.ja.json")  # 日语词表 / 日本語語彙
        tgt_vocab = load_vocab(CH13_PROCESSED_DIR / "vocab.en.json")  # 英语词表 / 英語語彙
        src_lines = load_token_lines(CH13_PROCESSED_DIR / "train.ja.tok")[: args.max_train_examples]  # 训练源 / 訓練入力
        tgt_lines = load_token_lines(CH13_PROCESSED_DIR / "train.en.tok")[: args.max_train_examples]  # 训练目标 / 訓練目標
        train_ref = [" ".join(line) for line in tgt_lines]  # 训练参考 / 訓練参照訳
        dev_src = load_token_lines(CH13_PROCESSED_DIR / "dev.ja.tok")  # 开发源 / 開発入力
        dev_tgt = load_token_lines(CH13_PROCESSED_DIR / "dev.en.tok")  # 开发目标 / 開発目標
        dev_ref = [" ".join(line) for line in dev_tgt]  # 开发参考 / 開発参照訳
        dataset = TranslationDataset(src_lines, tgt_lines, src_vocab, tgt_vocab)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, collate_fn=collate_translation)  # loader / loader
        dev_loader = DataLoader(TranslationDataset(dev_src, dev_tgt, src_vocab, tgt_vocab), batch_size=args.batch_size, shuffle=False, collate_fn=collate_translation)  # dev loader / dev loader
        config = {"emb_size": 256, "nhead": 4, "num_layers": 3, "dim_feedforward": 512}  # 模型配置 / モデル設定
        model = TransformerMT(len(src_vocab), len(tgt_vocab), **config).to(device)  # 创建模型 / モデルを作る
        if distributed:
            model = DDP(model, device_ids=[local_rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 优化器 / 最適化器
        loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])  # loss函数 / loss関数
        writer = SummaryWriter(args.log_dir) if is_main_process() else None  # 创建writer / writerを作る
        for epoch in range(1, args.epochs + 1):  # epoch循环 / epochループ
            if sampler is not None:
                sampler.set_epoch(epoch)
            train_loss = train_epoch(model, loader, optimizer, loss_fn, device)  # 训练一轮 / 1epoch学習
            dev_loss = evaluate_loss(model, dev_loader, loss_fn, device)  # 开发loss / 開発loss
            if is_main_process():
                eval_model = model.module if hasattr(model, "module") else model
                train_bleu, _train_preds = evaluate_bleu_for_beam(eval_model, src_lines, train_ref, src_vocab, tgt_vocab, beam_size=1, max_examples=args.max_train_bleu_examples, device=device)  # 训练BLEU / 訓練BLEU
                dev_bleu, _dev_preds = evaluate_bleu_for_beam(eval_model, dev_src, dev_ref, src_vocab, tgt_vocab, beam_size=1, max_examples=args.max_dev_examples, device=device)  # 开发BLEU / 開発BLEU
                writer.add_scalar("train/loss", train_loss, epoch)  # 记录训练loss / 訓練lossを記録
                writer.add_scalar("train/bleu", train_bleu, epoch)  # 记录训练BLEU / 訓練BLEUを記録
                writer.add_scalar("dev/loss", dev_loss, epoch)  # 记录开发loss / 開発lossを記録
                writer.add_scalar("dev/bleu", dev_bleu, epoch)  # 记录开发BLEU / 開発BLEUを記録
                print(f"epoch {epoch:02d}: train_loss={train_loss:.6f}, train_bleu={train_bleu:.2f}, dev_loss={dev_loss:.6f}, dev_bleu={dev_bleu:.2f}")  # 输出进度 / 進捗を出力
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        if writer is not None:
            writer.close()  # 关闭writer / writerを閉じる
    finally:
        cleanup_distributed()


if __name__ == "__main__":  # 直接运行 / 直接実行
    main()  # 调用主函数 / メイン関数を呼ぶ


# AIX実行結果メモ (2026-07-19, log: ~/100knock/logs/ch14_knock96_limited_g24.log)
# limited TensorBoard run used one epoch and logged train/dev loss and BLEU.
# epoch 01: train_loss=8.148319, train_bleu=0.00, dev_loss=6.934826, dev_bleu=0.01.
# TensorBoard event saved under Chapter 14/outputs/tensorboard_limited/.
# Dev BLEU stayed near zero, so the run mainly confirms logging/monitoring pipeline rather than model quality.
