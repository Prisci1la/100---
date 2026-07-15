'''
knock98.py: ドメイン適応 / ドメイン適応

JESCやJParaCrawlなどの追加対訳データを用いて、KFTTモデルを継続学習する。
/ JESCやJParaCrawlなどの追加対訳データを用いて、KFTTモデルを継続学習する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # PyTorch / PyTorch
from torch import nn  # 神经网络模块 / ニューラルネットワーク
from torch.utils.data import DataLoader  # DataLoader / DataLoader

from chapter14_utils import BASE_CHECKPOINT, CHECKPOINT_DIR, TranslationDataset, collate_translation, get_device, load_checkpoint, read_parallel_text, save_checkpoint, train_epoch  # 共用工具 / 共通ツール


def main():  # 主函数 / メイン関数
    parser = argparse.ArgumentParser(description="knock98: domain adaptation")  # 参数解析 / 引数解析
    parser.add_argument("--checkpoint", default=str(BASE_CHECKPOINT), help="base checkpoint")  # 基础checkpoint / 基礎checkpoint
    parser.add_argument("--ja-path", required=True, help="additional Japanese corpus path")  # 追加日语语料 / 追加日本語コーパス
    parser.add_argument("--en-path", required=True, help="additional English corpus path")  # 追加英语语料 / 追加英語コーパス
    parser.add_argument("--output", default=str(CHECKPOINT_DIR / "domain_adapted_mt.pt"), help="output checkpoint")  # 输出checkpoint / 出力checkpoint
    parser.add_argument("--epochs", type=int, default=1, help="epochs")  # epoch数 / epoch数
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")  # batch大小 / batchサイズ
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")  # 学习率 / 学習率
    parser.add_argument("--max-lines", type=int, default=None, help="limit additional corpus lines for quick check")  # 行数上限 / 行数上限
    args = parser.parse_args()  # 解析 / 解析する
    device = get_device()  # 设备 / デバイス
    if device.type != "cuda":  # 题目要求GPU训练 / 課題はGPU学習を要求する
        raise RuntimeError("knock98 requires GPU training. CUDA is not available.")  # 报错 / エラー
    model, src_vocab, tgt_vocab, config = load_checkpoint(args.checkpoint, device)  # 读取模型 / モデルを読む
    ja_lines, en_lines = read_parallel_text(args.ja_path, args.en_path, args.max_lines)  # 读取追加语料 / 追加コーパスを読む
    loader = DataLoader(TranslationDataset(ja_lines, en_lines, src_vocab, tgt_vocab), batch_size=args.batch_size, shuffle=True, collate_fn=collate_translation)  # loader / loader
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 优化器 / 最適化器
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])  # loss / loss
    for epoch in range(1, args.epochs + 1):  # epoch循环 / epochループ
        loss = train_epoch(model, loader, optimizer, loss_fn, device)  # 训练 / 学習
        print(f"epoch {epoch:02d}: loss={loss:.6f}")  # 输出 / 出力
    save_checkpoint(args.output, model, src_vocab, tgt_vocab, config)  # 保存模型 / モデルを保存


if __name__ == "__main__":  # 直接运行 / 直接実行
    main()  # 调用主函数 / メイン関数を呼ぶ

