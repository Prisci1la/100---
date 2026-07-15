'''
knock97.py: ハイパー・パラメータの調整 / ハイパー・パラメータの調整

batch size、learning rate、Optimizerを変えながら開発BLEUを比較する。
/ batch size、learning rate、Optimizerを変えながら開発BLEUを比較する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ
import itertools  # 组合生成 / 組み合わせ生成

import torch  # PyTorch / PyTorch
from torch import nn  # 神经网络模块 / ニューラルネットワーク
from torch.utils.data import DataLoader  # DataLoader / DataLoader

from chapter14_utils import CHECKPOINT_DIR, CH13_PROCESSED_DIR, OUTPUT_DIR, TransformerMT, TranslationDataset, collate_translation, evaluate_bleu_for_beam, get_device, load_token_lines, load_vocab, save_checkpoint, train_epoch, write_json  # 共用工具 / 共通ツール


def make_optimizer(name, parameters, lr):  # 创建优化器 / 最適化器を作る
    name = name.lower()  # 标准化名称 / 名前を正規化
    if name == "adam":  # Adam / Adam
        return torch.optim.Adam(parameters, lr=lr)  # 返回Adam / Adamを返す
    if name == "radam":  # RAdam / RAdam
        return torch.optim.RAdam(parameters, lr=lr)  # 返回RAdam / RAdamを返す
    if name != "adamw":  # 检查名称 / 名前を確認する
        raise ValueError(f"unsupported optimizer: {name}")  # 报错 / エラー
    return torch.optim.AdamW(parameters, lr=lr)  # 默认AdamW / 既定はAdamW


def main():  # 主函数 / メイン関数
    parser = argparse.ArgumentParser(description="knock97: hyperparameter tuning")  # 参数解析 / 引数解析
    parser.add_argument("--batch-sizes", default="32,64", help="comma separated batch sizes")  # batch候选 / batch候補
    parser.add_argument("--lrs", default="1e-3,1e-4,1e-5,1e-6", help="comma separated learning rates")  # lr候选 / lr候補
    parser.add_argument("--optimizers", default="adam,adamw,radam", help="comma separated optimizer names")  # optimizer候选 / optimizer候補
    parser.add_argument("--epochs", type=int, default=1, help="epochs per trial")  # 每次epoch / 各試行epoch
    parser.add_argument("--max-train-examples", type=int, default=None, help="limit train examples for quick check")  # 训练上限 / 訓練上限
    parser.add_argument("--max-dev-examples", type=int, default=None, help="limit dev examples for quick check")  # 开发上限 / 開発上限
    args = parser.parse_args()  # 解析 / 解析する
    device = get_device()  # 设备 / デバイス
    if device.type != "cuda":  # 题目要求GPU训练 / 課題はGPU学習を要求する
        raise RuntimeError("knock97 requires GPU training. CUDA is not available.")  # 报错 / エラー
    src_vocab = load_vocab(CH13_PROCESSED_DIR / "vocab.ja.json")  # 源词表 / 入力語彙
    tgt_vocab = load_vocab(CH13_PROCESSED_DIR / "vocab.en.json")  # 目标词表 / 目標語彙
    src_lines = load_token_lines(CH13_PROCESSED_DIR / "train.ja.tok")[: args.max_train_examples]  # 训练源 / 訓練入力
    tgt_lines = load_token_lines(CH13_PROCESSED_DIR / "train.en.tok")[: args.max_train_examples]  # 训练目标 / 訓練目標
    dev_src = load_token_lines(CH13_PROCESSED_DIR / "dev.ja.tok")  # 开发源 / 開発入力
    dev_ref = [" ".join(line) for line in load_token_lines(CH13_PROCESSED_DIR / "dev.en.tok")]  # 开发参考 / 開発参照
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]  # batch列表 / batchリスト
    lrs = [float(x) for x in args.lrs.split(",")]  # lr列表 / lrリスト
    optimizers = [x.strip() for x in args.optimizers.split(",")]  # optimizer列表 / optimizerリスト
    results = []  # 保存结果 / 結果を保存する
    best_result = None  # 最佳结果 / 最良結果
    for batch_size, lr, opt_name in itertools.product(batch_sizes, lrs, optimizers):  # 遍历组合 / 組み合わせを走査
        config = {"emb_size": 256, "nhead": 4, "num_layers": 3, "dim_feedforward": 512}  # 模型配置 / モデル設定
        model = TransformerMT(len(src_vocab), len(tgt_vocab), **config).to(device)  # 模型 / モデル
        loader = DataLoader(TranslationDataset(src_lines, tgt_lines, src_vocab, tgt_vocab), batch_size=batch_size, shuffle=True, collate_fn=collate_translation)  # loader / loader
        optimizer = make_optimizer(opt_name, model.parameters(), lr)  # 优化器 / 最適化器
        loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])  # loss / loss
        for _epoch in range(args.epochs):  # 训练指定epoch / 指定epoch学習
            loss = train_epoch(model, loader, optimizer, loss_fn, device)  # 学习 / 学習
        bleu, _preds = evaluate_bleu_for_beam(model, dev_src, dev_ref, src_vocab, tgt_vocab, 1, args.max_dev_examples, device)  # BLEU / BLEU
        result = {"batch_size": batch_size, "lr": lr, "optimizer": opt_name, "loss": loss, "dev_bleu": bleu}  # 结果字典 / 結果辞書
        results.append(result)  # 保存 / 保存
        if best_result is None or bleu > best_result["dev_bleu"]:  # 更新最佳 / 最良を更新
            best_result = result  # 保存最佳结果 / 最良結果を保存
            save_checkpoint(CHECKPOINT_DIR / "knock97_best_mt.pt", model, src_vocab, tgt_vocab, config)  # 保存最佳模型 / 最良モデルを保存
            write_json(OUTPUT_DIR / "knock97_best_hparams.json", best_result)  # 保存最佳参数 / 最良パラメータを保存
        print(result)  # 输出 / 出力
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 输出目录 / 出力ディレクトリ
    write_json(OUTPUT_DIR / "knock97_hparam_results.json", results)  # 保存结果 / 結果を保存
    print(f"best: {best_result}")  # 输出最佳 / 最良を出力
    if best_result is not None and best_result["dev_bleu"] < 10:  # BLEU目标检查 / BLEU目標確認
        print("warning: best dev BLEU is below 10. Try more epochs or a larger model.")  # 提示 / ヒント


if __name__ == "__main__":  # 直接运行 / 直接実行
    main()  # 调用主函数 / メイン関数を呼ぶ

