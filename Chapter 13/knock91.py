'''
knock91.py: 機械翻訳モデルの訓練 / 機械翻訳モデルの訓練

90で準備したKFTTを用いて、PyTorch Transformer翻訳モデルを訓練する。
/ 90で準備したKFTTを用いて、PyTorch Transformer翻訳モデルを訓練する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # PyTorch / PyTorch
from torch import nn  # 神经网络模块 / ニューラルネットワーク
from torch.utils.data import DataLoader  # DataLoader / DataLoader

from chapter13_utils import CHECKPOINT_DIR, PROCESSED_DIR, TransformerMT, TranslationDataset, collate_translation, get_device, load_token_lines, load_vocab, save_checkpoint, train_epoch  # 共用工具 / 共通ツール


def main():  # 主函数 / メイン関数
    parser = argparse.ArgumentParser(description="knock91: train Transformer MT")  # 参数解析 / 引数解析
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")  # epoch数 / epoch数
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")  # batch大小 / batchサイズ
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")  # 学习率 / 学習率
    parser.add_argument("--emb-size", type=int, default=256, help="embedding size")  # embedding维度 / embedding次元
    parser.add_argument("--nhead", type=int, default=4, help="attention heads")  # head数 / head数
    parser.add_argument("--num-layers", type=int, default=3, help="Transformer layers")  # 层数 / 層数
    parser.add_argument("--max-train-examples", type=int, default=None, help="limit train examples")  # 样本上限 / サンプル上限
    args = parser.parse_args()  # 解析 / 解析する
    device = get_device()  # 获取设备 / デバイス取得
    if device.type != "cuda":  # 题目要求使用GPU训练 / 課題はGPU学習を要求する
        raise RuntimeError("knock91 requires GPU training. CUDA is not available.")
    src_vocab = load_vocab(PROCESSED_DIR / "vocab.ja.json")  # 读取日语词表 / 日本語語彙を読む
    tgt_vocab = load_vocab(PROCESSED_DIR / "vocab.en.json")  # 读取英语词表 / 英語語彙を読む
    src_lines = load_token_lines(PROCESSED_DIR / "train.ja.tok")[: args.max_train_examples]  # 读取源 / 入力を読む
    tgt_lines = load_token_lines(PROCESSED_DIR / "train.en.tok")[: args.max_train_examples]  # 读取目标 / 目標を読む
    dataset = TranslationDataset(src_lines, tgt_lines, src_vocab, tgt_vocab)  # 创建Dataset / Datasetを作る
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_translation)  # 创建DataLoader / DataLoaderを作る
    config = {"emb_size": args.emb_size, "nhead": args.nhead, "num_layers": args.num_layers, "dim_feedforward": args.emb_size * 2}  # 模型配置 / モデル設定
    model = TransformerMT(len(src_vocab), len(tgt_vocab), **config).to(device)  # 创建模型 / モデルを作る
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 优化器 / 最適化器
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_vocab["<pad>"])  # 损失函数 / 損失関数
    print("=" * 50)  # 分隔线 / 区切り線
    print("Knock 91: Train Transformer MT")  # 标题 / タイトル
    print("=" * 50)  # 分隔线 / 区切り線
    for epoch in range(1, args.epochs + 1):  # epoch循环 / epochループ
        loss = train_epoch(model, loader, optimizer, loss_fn, device)  # 学习一轮 / 1epoch学習
        print(f"epoch {epoch:02d}: loss={loss:.6f}")  # 输出loss / lossを出力
    save_checkpoint(CHECKPOINT_DIR / "transformer_mt.pt", model, src_vocab, tgt_vocab, config)  # 保存模型 / モデルを保存する


if __name__ == "__main__":  # 直接运行 / 直接実行
    main()  # 调用主函数 / メイン関数を呼ぶ

