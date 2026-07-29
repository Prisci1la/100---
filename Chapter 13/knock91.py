'''
knock91.py: 機械翻訳モデルの訓練 / 機械翻訳モデルの訓練

90で準備したKFTTを用いて、PyTorch Transformer翻訳モデルを訓練する。
/ 90で準備したKFTTを用いて、PyTorch Transformer翻訳モデルを訓練する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ
import json  # JSON读取库 / JSON読込ライブラリ
import math  # 数学函数 / 数学関数
import os  # 环境变量 / 環境変数
from pathlib import Path  # 路径处理 / パス処理

import torch  # PyTorch / PyTorch
import torch.distributed as dist  # 分布式训练 / 分散学習
from torch import nn  # 神经网络模块 / ニューラルネットワーク
from torch.utils.data import DataLoader, Dataset  # 数据工具 / データツール
from torch.utils.data.distributed import DistributedSampler  # DDP sampler / DDP sampler
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP / DDP
from tqdm.auto import tqdm  # 进度条 / 進捗バー


BASE_DIR = Path(__file__).resolve().parent  # 当前第13回目录 / 現在の第13回ディレクトリ
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # 前处理数据目录 / 前処理データディレクトリ
CHECKPOINT_DIR = BASE_DIR / "checkpoints"  # 模型保存目录 / モデル保存ディレクトリ
BOS = "<bos>"  # 开始token / 開始token
EOS = "<eos>"  # 结束token / 終了token
UNK = "<unk>"  # 未知token / 未知token


def get_device():  # 获取可用设备 / 利用可能なデバイスを取得する
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有GPU则用CUDA / GPUがあればCUDAを使う


def setup_distributed():  # 初始化torchrun分布式环境 / torchrun分散環境を初期化する
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:  # 非torchrun环境 / torchrun環境でない場合
        return False, 0, 0, 1  # 返回单进程设置 / 単一process設定を返す
    rank = int(os.environ["RANK"])  # 全局进程编号 / global process番号
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 当前节点GPU编号 / 現在nodeのGPU番号
    world_size = int(os.environ["WORLD_SIZE"])  # 总进程数 / 総process数
    if torch.cuda.is_available():  # CUDA可用时 / CUDAが利用可能な場合
        torch.cuda.set_device(local_rank)  # 绑定当前进程的GPU / 現在processのGPUを設定する
    if not dist.is_initialized():  # 尚未初始化时 / 未初期化の場合
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")  # 初始化进程组 / process groupを初期化する
    return True, rank, local_rank, world_size  # 返回分布式设置 / 分散設定を返す


def cleanup_distributed():  # 关闭分布式环境 / 分散環境を閉じる
    if dist.is_available() and dist.is_initialized():  # 进程组存在时 / process groupが存在する場合
        dist.destroy_process_group()  # 释放分布式资源 / 分散resourceを解放する


def is_main_process():  # 是否主进程 / 主プロセスかどうか
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0  # 单进程或rank 0为主进程 / 単一processまたはrank 0を主processとする


def unwrap_model(model):  # 取得DDP内部模型 / DDP内部モデルを取得する
    return model.module if hasattr(model, "module") else model  # DDP时取内部模型 / DDPなら内部modelを取る


def load_token_lines(path):  # 读取tokenized文件 / tokenizedファイルを読む
    return [line.strip().split() for line in Path(path).open(encoding="utf-8") if line.strip()]  # 返回非空token列 / 空でないtoken列を返す


def load_vocab(path):  # 读取词表 / 語彙を読み込む
    return json.loads(Path(path).read_text(encoding="utf-8"))  # 从UTF-8 JSON恢复词表 / UTF-8 JSONから語彙を復元する


def encode(tokens, vocab):  # token列转ID列 / token列をID列へ変換する
    return [vocab[BOS]] + [vocab.get(token, vocab[UNK]) for token in tokens] + [vocab[EOS]]  # 添加BOS/EOS并替换未知词 / BOS・EOSを付け未知語を置換する


class TranslationDataset(Dataset):  # 翻译数据集 / 翻訳データセット
    def __init__(self, src_lines, tgt_lines, src_vocab, tgt_vocab):  # 初始化数据集 / datasetを初期化する
        self.src = [encode(line, src_vocab) for line in src_lines]  # 编码全部源句 / 全入力文を符号化する
        self.tgt = [encode(line, tgt_vocab) for line in tgt_lines]  # 编码全部目标句 / 全目標文を符号化する

    def __len__(self):  # 返回样本数 / sample数を返す
        return len(self.src)  # 以源句数作为长度 / 入力文数を長さとする

    def __getitem__(self, index):  # 取得一个句对 / 1つの対訳を取得する
        return torch.tensor(self.src[index]), torch.tensor(self.tgt[index])  # 转成Tensor返回 / Tensorへ変換して返す


def collate_translation(batch, pad_id=0):  # padding batch / batchをpaddingする
    src, tgt = zip(*batch)  # 分离源句和目标句 / 入力文と目標文を分ける
    src_pad = nn.utils.rnn.pad_sequence(src, padding_value=pad_id)  # 对源句做padding / 入力文をpaddingする
    tgt_pad = nn.utils.rnn.pad_sequence(tgt, padding_value=pad_id)  # 对目标句做padding / 目標文をpaddingする
    return src_pad, tgt_pad  # seq_len x batch / seq_len x batch


class PositionalEncoding(nn.Module):  # 位置编码 / 位置符号化
    def __init__(self, emb_size, dropout=0.1, max_len=5000):  # 初始化位置编码 / 位置符号化を初期化する
        super().__init__()  # 初始化父类 / 親classを初期化する
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)  # 计算正弦波频率 / 正弦波の周波数を計算する
        pos = torch.arange(0, max_len).reshape(max_len, 1)  # 创建位置编号 / 位置番号を作る
        pe = torch.zeros(max_len, emb_size)  # 初始化位置编码矩阵 / 位置符号化行列を初期化する
        pe[:, 0::2] = torch.sin(pos * den)  # 偶数维使用sin / 偶数次元にsinを使う
        pe[:, 1::2] = torch.cos(pos * den)  # 奇数维使用cos / 奇数次元にcosを使う
        self.dropout = nn.Dropout(dropout)  # 创建Dropout层 / Dropout層を作る
        self.register_buffer("pe", pe.unsqueeze(1))  # 注册无需训练的位置编码 / 学習不要の位置符号化を登録する

    def forward(self, x):  # 前向计算 / 順伝播
        return self.dropout(x + self.pe[: x.size(0)])  # 加位置编码后应用Dropout / 位置符号化を足してDropoutする


class TransformerMT(nn.Module):  # Transformer翻译模型 / Transformer翻訳モデル
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_size=256, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1):  # 初始化模型 / modelを初期化する
        super().__init__()  # 初始化父类 / 親classを初期化する
        self.src_embedding = nn.Embedding(src_vocab_size, emb_size, padding_idx=0)  # 源语言embedding / 入力言語embedding
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=0)  # 目标语言embedding / 目標言語embedding
        self.positional_encoding = PositionalEncoding(emb_size, dropout)  # 创建位置编码 / 位置符号化を作る
        self.transformer = nn.Transformer(emb_size, nhead, num_layers, num_layers, dim_feedforward, dropout)  # 创建Transformer主体 / Transformer本体を作る
        self.generator = nn.Linear(emb_size, tgt_vocab_size)  # 将隐藏状态映射到词表 / 隠れ状態を語彙へ写像する
        self.emb_size = emb_size  # 保存embedding维度 / embedding次元を保存する

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):  # 前向计算 / 順伝播
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.emb_size))  # 计算源语言表示 / 入力言語表現を計算する
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.emb_size))  # 计算目标语言表示 / 目標言語表現を計算する
        tgt_mask = create_subsequent_mask(tgt.size(0), tgt.device)  # 屏蔽decoder未来位置 / decoderの未来位置を隠す
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)  # 执行Transformer / Transformerを実行する
        return self.generator(out)  # 输出各token的logits / 各tokenのlogitsを出力する


def create_padding_mask(seq, pad_id=0):  # 创建padding mask / padding maskを作る
    return (seq == pad_id).transpose(0, 1)  # 转成batch x seq_len / batch x seq_lenへ変換する


def create_subsequent_mask(size, device):  # 创建decoder未来mask / decoder未来maskを作る
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)  # 上三角位置设为负无穷 / 上三角位置を負の無限大にする


def train_epoch(model, loader, optimizer, loss_fn, device, reduce_distributed=True):  # 训练一轮 / 1epoch学習する
    model.train()  # 切换训练模式 / 学習modeへ切り替える
    total_loss = 0.0  # 初始化累计损失 / 累積lossを初期化する
    for src, tgt in tqdm(loader, desc="train", leave=False):  # 遍历mini-batch / mini-batchを走査する
        src, tgt = src.to(device), tgt.to(device)  # 移到训练设备 / 学習deviceへ移す
        tgt_input = tgt[:-1, :]  # 去掉最后token作为decoder输入 / 最終tokenを除いてdecoder入力にする
        tgt_out = tgt[1:, :]  # 去掉BOS作为正确答案 / BOSを除いて正解にする
        optimizer.zero_grad()  # 清空上一批梯度 / 前batchの勾配を消す
        logits = model(src, tgt_input, create_padding_mask(src), create_padding_mask(tgt_input), create_padding_mask(src))  # 前向计算 / 順伝播
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))  # 计算交叉熵损失 / cross entropy lossを計算する
        loss.backward()  # 反向传播 / 逆伝播
        optimizer.step()  # 更新参数 / parameterを更新する
        total_loss += loss.item()  # 累加当前损失 / 現在のlossを加算する
    stats = torch.tensor([total_loss, len(loader)], dtype=torch.float64, device=device)  # 汇总损失和batch数 / lossとbatch数をまとめる
    if reduce_distributed and dist.is_available() and dist.is_initialized():  # 需要跨进程汇总时 / process間集計が必要な場合
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)  # 汇总全部进程的统计量 / 全processの統計量を集約する
    return (stats[0] / stats[1].clamp(min=1)).item()  # 返回平均损失 / 平均lossを返す


def save_checkpoint(path, model, src_vocab, tgt_vocab, config):  # 保存checkpoint / checkpointを保存する
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # 创建checkpoint目录 / checkpointディレクトリを作る
    torch.save({"model_state": unwrap_model(model).state_dict(), "src_vocab": src_vocab, "tgt_vocab": tgt_vocab, "config": config}, path)  # 保存权重、词表和配置 / 重み・語彙・設定を保存する


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
    distributed, rank, local_rank, world_size = setup_distributed()  # DDP初期化 / DDP初始化
    try:  # 确保最终释放分布式资源 / 最後に分散resourceを必ず解放する
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu") if distributed else get_device()  # 获取设备 / デバイス取得
        if device.type != "cuda":  # 题目要求使用GPU训练 / 課題はGPU学習を要求する
            raise RuntimeError("knock91 requires GPU training. CUDA is not available.")  # 无CUDA时明确停止 / CUDAがなければ明示的に停止する
        src_vocab = load_vocab(PROCESSED_DIR / "vocab.ja.json")  # 读取日语词表 / 日本語語彙を読む
        tgt_vocab = load_vocab(PROCESSED_DIR / "vocab.en.json")  # 读取英语词表 / 英語語彙を読む
        src_lines = load_token_lines(PROCESSED_DIR / "train.ja.tok")[: args.max_train_examples]  # 读取源 / 入力を読む
        tgt_lines = load_token_lines(PROCESSED_DIR / "train.en.tok")[: args.max_train_examples]  # 读取目标 / 目標を読む
        dataset = TranslationDataset(src_lines, tgt_lines, src_vocab, tgt_vocab)  # 创建Dataset / Datasetを作る
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None  # sampler / sampler
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, collate_fn=collate_translation)  # 创建DataLoader / DataLoaderを作る
        config = {"emb_size": args.emb_size, "nhead": args.nhead, "num_layers": args.num_layers, "dim_feedforward": args.emb_size * 2}  # 模型配置 / モデル設定
        model = TransformerMT(len(src_vocab), len(tgt_vocab), **config).to(device)  # 创建模型 / モデルを作る
        if distributed:  # 分布式训练时 / 分散学習の場合
            model = DDP(model, device_ids=[local_rank])  # DDP包装 / DDPで包む
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 优化器 / 最適化器
        loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])  # 损失函数 / 損失関数
        if is_main_process():  # 仅由主进程输出设置 / 主processだけ設定を出力する
            print("=" * 50)  # 分隔线 / 区切り線
            print("Knock 91: Train Transformer MT")  # 标题 / タイトル
            print("=" * 50)  # 分隔线 / 区切り線
            print(f"device: {device}, train: {len(src_lines)}, batch_size_per_gpu: {args.batch_size}, epochs: {args.epochs}, world_size: {world_size}")  # 输出设置 / 設定を出力
        for epoch in range(1, args.epochs + 1):  # epoch循环 / epochループ
            if sampler is not None:  # 使用分布式sampler时 / 分散samplerを使う場合
                sampler.set_epoch(epoch)  # 每轮改变shuffle种子 / epochごとにshuffle seedを変える
            loss = train_epoch(model, loader, optimizer, loss_fn, device)  # 学习一轮 / 1epoch学習
            if is_main_process():  # 仅由主进程输出损失 / 主processだけlossを出力する
                print(f"epoch {epoch:02d}: loss={loss:.6f}")  # 输出loss / lossを出力
        if is_main_process():  # 仅由主进程保存 / 主processだけ保存する
            save_checkpoint(CHECKPOINT_DIR / "transformer_mt.pt", model, src_vocab, tgt_vocab, config)  # 保存模型 / モデルを保存する
    finally:  # 正常或异常退出都执行 / 正常終了でも例外でも実行する
        cleanup_distributed()  # 关闭分布式进程组 / 分散process groupを閉じる


if __name__ == "__main__":  # 直接运行 / 直接実行
    main()  # 调用主函数 / メイン関数を呼ぶ


# AIX実行結果メモ (2026-07-18, logs: ch13_knock91*.log)
# first run: device=cuda:0, train=440286, batch_size_per_gpu=64, epochs=5, world_size=8.
# failure: CUDA OOM around 52% of epoch 1 on GPU6; attempted allocation was about 4.02GiB.
# retry: batch_size_per_gpu=16, epochs=5, world_size=8; each epoch had 3440 distributed train steps.
# epoch losses: 01=6.120504; 02=5.229440; 03=4.842080; 04=4.582713; 05=4.385998.
# checkpoint saved to Chapter 13/checkpoints/transformer_mt.pt.
