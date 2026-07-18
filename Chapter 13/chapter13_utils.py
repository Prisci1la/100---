'''
chapter13_utils.py: 第13回共用工具 / 第13回の共通ユーティリティ

KFTTの取得、前処理、語彙構築、PyTorch Transformer翻訳、BLEU計測をまとめる。
/ KFTT取得、前処理、語彙構築、PyTorch Transformer翻訳、BLEU計測をまとめる。
'''

from __future__ import annotations  # 延迟类型注解的求值 / 型注釈の評価を遅延する

import json  # JSON保存库 / JSON保存ライブラリ
import math  # 数学函数 / 数学関数
import tarfile  # tar.gz展开库 / tar.gz展開ライブラリ
import urllib.request  # 下载库 / ダウンロードライブラリ
from collections import Counter  # 词频统计 / 語彙頻度集計
from pathlib import Path  # 路径处理 / パス処理

import sacrebleu  # BLEU计算库 / BLEU計算ライブラリ
import torch  # PyTorch / PyTorch
from torch import nn  # 神经网络模块 / ニューラルネットワークモジュール
from torch.utils.data import DataLoader, Dataset  # 数据工具 / データツール
from tqdm.auto import tqdm  # 进度条 / 進捗バー


BASE_DIR = Path(__file__).resolve().parent  # 当前第13回目录 / 現在の第13回ディレクトリ
KFTT_URL = "https://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz"  # KFTT下载地址 / KFTTダウンロード先
RAW_DIR = BASE_DIR / "data" / "raw"  # 原始数据目录 / 生データディレクトリ
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # 前处理数据目录 / 前処理データディレクトリ
CHECKPOINT_DIR = BASE_DIR / "checkpoints"  # 模型保存目录 / モデル保存ディレクトリ
OUTPUT_DIR = BASE_DIR / "outputs"  # 输出目录 / 出力ディレクトリ
PAD = "<pad>"  # padding token / padding token
BOS = "<bos>"  # 开始token / 開始token
EOS = "<eos>"  # 结束token / 終了token
UNK = "<unk>"  # 未知token / 未知token
SPECIALS = [PAD, BOS, EOS, UNK]  # 特殊token列表 / 特殊token一覧


def get_device():  # 获取可用设备 / 利用可能なデバイスを取得する
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有GPU就用CUDA / GPUがあればCUDAを使う


def download_kftt(url=KFTT_URL, archive_path=RAW_DIR / "kftt-data-1.0.tar.gz"):  # 下载KFTT / KFTTをダウンロードする
    archive_path = Path(archive_path)  # 转换为Path / Pathへ変換する
    archive_path.parent.mkdir(parents=True, exist_ok=True)  # 创建目录 / ディレクトリを作る
    if not archive_path.exists():  # 文件不存在时下载 / ファイルがない場合はダウンロードする
        urllib.request.urlretrieve(url, archive_path)  # 下载文件 / ファイルをダウンロードする
    return archive_path  # 返回压缩包路径 / アーカイブパスを返す


def extract_kftt(archive_path=RAW_DIR / "kftt-data-1.0.tar.gz", raw_dir=RAW_DIR):  # 展开KFTT / KFTTを展開する
    archive_path = Path(archive_path)  # 转换为Path / Pathへ変換する
    raw_dir = Path(raw_dir)  # 转换为Path / Pathへ変換する
    target_dir = raw_dir / "kftt-data-1.0"  # 展开目标目录 / 展開先ディレクトリ
    if not target_dir.exists():  # 未展开时处理 / 未展開なら処理する
        with tarfile.open(archive_path, "r:gz") as tar:  # 打开tar.gz / tar.gzを開く
            tar.extractall(raw_dir)  # 展开文件 / ファイルを展開する
    return target_dir  # 返回数据目录 / データディレクトリを返す


def find_parallel_files(kftt_dir, split):  # 查找并行语料文件 / 対訳ファイルを探す
    kftt_dir = Path(kftt_dir)  # 转换为Path / Pathへ変換する
    ja_candidates = list(kftt_dir.rglob(f"kyoto-{split}.ja")) + list(kftt_dir.rglob(f"{split}.ja"))  # 日语候选 / 日本語候補
    en_candidates = list(kftt_dir.rglob(f"kyoto-{split}.en")) + list(kftt_dir.rglob(f"{split}.en"))  # 英语候选 / 英語候補
    if not ja_candidates or not en_candidates:  # 找不到时抛错 / 見つからない場合はエラー
        raise FileNotFoundError(f"Could not find KFTT {split} files under {kftt_dir}")  # 明确报错 / 明確にエラーにする
    return ja_candidates[0], en_candidates[0]  # 返回第一组候选 / 最初の候補を返す


def get_ja_tagger():  # 创建日语分词器 / 日本語分かち書き器を作る
    try:  # 优先使用fugashi / fugashiを優先する
        import fugashi  # 导入fugashi / fugashiを導入する
        return fugashi.Tagger()  # 返回Tagger / Taggerを返す
    except Exception as exc:  # 失败时明确报错 / 失敗時は明確にエラー
        raise RuntimeError("Japanese morphological tokenizer is required. Install fugashi and unidic-lite.") from exc


def tokenize_ja(text, tagger=None):  # 日语分词 / 日本語を分かち書きする
    if tagger is None:  # 没有tagger时 / taggerがない場合
        tagger = get_ja_tagger()  # 使用形态素分词器 / 形態素解析器を使う
    return [word.surface for word in tagger(text.strip())]  # fugashi分词 / fugashiで分かち書きする


def tokenize_en(text):  # 英语tokenize / 英語tokenize
    return text.strip().lower().split()  # 简单按空格切分 / 簡単に空白で分ける


def preprocess_split(kftt_dir, split, out_dir=PROCESSED_DIR, max_lines=None):  # 前处理一个split / 1つのsplitを前処理する
    out_dir = Path(out_dir)  # 转换为Path / Pathへ変換する
    out_dir.mkdir(parents=True, exist_ok=True)  # 创建目录 / ディレクトリを作る
    ja_path, en_path = find_parallel_files(kftt_dir, split)  # 查找原始文件 / 元ファイルを探す
    tagger = get_ja_tagger()  # 创建日语tagger / 日本語taggerを作る
    out_ja = out_dir / f"{split}.ja.tok"  # 输出日语文件 / 日本語出力ファイル
    out_en = out_dir / f"{split}.en.tok"  # 输出英语文件 / 英語出力ファイル
    count = 0  # 行数计数 / 行数カウント
    with ja_path.open(encoding="utf-8") as ja_file, en_path.open(encoding="utf-8") as en_file, out_ja.open("w", encoding="utf-8") as ja_out, out_en.open("w", encoding="utf-8") as en_out:  # 打开文件 / ファイルを開く
        for ja_line, en_line in tqdm(zip(ja_file, en_file), desc=f"preprocess {split}", leave=False):  # 遍历平行句 / 対訳文を走査する
            ja_tokens = tokenize_ja(ja_line, tagger)  # 日语分词 / 日本語を分かち書きする
            en_tokens = tokenize_en(en_line)  # 英语分词 / 英語を分かち書きする
            if not ja_tokens or not en_tokens:  # 空句跳过 / 空文は飛ばす
                continue  # 下一行 / 次の行へ進む
            ja_out.write(" ".join(ja_tokens) + "\n")  # 写日语 / 日本語を書き込む
            en_out.write(" ".join(en_tokens) + "\n")  # 写英语 / 英語を書き込む
            count += 1  # 行数增加 / 行数を増やす
            if max_lines is not None and count >= max_lines:  # 达到上限 / 上限に達した場合
                break  # 停止 / 止める
    return out_ja, out_en, count  # 返回输出路径和行数 / 出力パスと行数を返す


def load_token_lines(path):  # 读取tokenized文件 / tokenizedファイルを読む
    return [line.strip().split() for line in Path(path).open(encoding="utf-8") if line.strip()]  # 返回token列表 / tokenリストを返す


def build_vocab(token_lines, max_size=50000, min_freq=2):  # 构建词表 / 語彙を構築する
    counter = Counter(token for line in token_lines for token in line)  # 统计词频 / 語彙頻度を数える
    vocab = {token: index for index, token in enumerate(SPECIALS)}  # 初始化特殊token / 特殊tokenを初期化する
    for token, freq in counter.most_common(max_size):  # 按频率遍历 / 頻度順に走査する
        if freq < min_freq or token in vocab:  # 低频或已存在跳过 / 低頻または既存なら飛ばす
            continue  # 下一词 / 次の語へ進む
        vocab[token] = len(vocab)  # 加入词表 / 語彙に追加する
    return vocab  # 返回词表 / 語彙を返す


def save_vocab(vocab, path):  # 保存词表 / 語彙を保存する
    Path(path).write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")  # 写JSON / JSONを書く


def load_vocab(path):  # 读取词表 / 語彙を読み込む
    return json.loads(Path(path).read_text(encoding="utf-8"))  # 读取JSON / JSONを読む


def encode(tokens, vocab):  # token列转ID列 / token列をID列へ変換する
    return [vocab[BOS]] + [vocab.get(token, vocab[UNK]) for token in tokens] + [vocab[EOS]]  # 添加BOS/EOS / BOS/EOSを付ける


def decode(ids, vocab):  # ID列转token列 / ID列をtoken列へ変換する
    inv_vocab = {idx: token for token, idx in vocab.items()}  # 反向词表 / 逆語彙
    tokens = []  # 初始化token列表 / tokenリストを初期化する
    for idx in ids:  # 遍历ID / IDを走査する
        token = inv_vocab.get(int(idx), UNK)  # ID转token / IDをtokenへ変換する
        if token == EOS:  # 到EOS停止 / EOSで止める
            break  # 结束 / 終了
        if token not in {BOS, PAD}:  # 排除特殊token / 特殊tokenを除く
            tokens.append(token)  # 保存token / tokenを保存する
    return tokens  # 返回token列 / token列を返す


class TranslationDataset(Dataset):  # 翻译数据集 / 翻訳データセット
    def __init__(self, src_lines, tgt_lines, src_vocab, tgt_vocab):  # 初始化 / 初期化する
        self.src = [encode(line, src_vocab) for line in src_lines]  # 编码源语言 / 入力言語を符号化する
        self.tgt = [encode(line, tgt_vocab) for line in tgt_lines]  # 编码目标语言 / 目標言語を符号化する

    def __len__(self):  # 返回样本数 / サンプル数を返す
        return len(self.src)  # 返回长度 / 長さを返す

    def __getitem__(self, index):  # 返回一个样本 / 1サンプルを返す
        return torch.tensor(self.src[index]), torch.tensor(self.tgt[index])  # 返回Tensor / Tensorを返す


def collate_translation(batch, pad_id=0):  # padding batch / batchをpaddingする
    src, tgt = zip(*batch)  # 拆分源和目标 / 入力と目標を分ける
    src_pad = nn.utils.rnn.pad_sequence(src, padding_value=pad_id)  # padding源 / 入力をpaddingする
    tgt_pad = nn.utils.rnn.pad_sequence(tgt, padding_value=pad_id)  # padding目标 / 目標をpaddingする
    return src_pad, tgt_pad  # 返回seq_len x batch / seq_len x batchを返す


class PositionalEncoding(nn.Module):  # 位置编码 / 位置符号化
    def __init__(self, emb_size, dropout=0.1, max_len=5000):  # 初始化 / 初期化する
        super().__init__()  # 父类初始化 / 親クラス初期化
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)  # 计算分母 / 分母を計算する
        pos = torch.arange(0, max_len).reshape(max_len, 1)  # 位置索引 / 位置index
        pe = torch.zeros(max_len, emb_size)  # 初始化矩阵 / 行列を初期化する
        pe[:, 0::2] = torch.sin(pos * den)  # 偶数维 / 偶数次元
        pe[:, 1::2] = torch.cos(pos * den)  # 奇数维 / 奇数次元
        self.dropout = nn.Dropout(dropout)  # Dropout / Dropout
        self.register_buffer("pe", pe.unsqueeze(1))  # 注册buffer / bufferとして登録する

    def forward(self, x):  # 前向计算 / 順伝播
        return self.dropout(x + self.pe[: x.size(0)])  # 加位置编码 / 位置符号化を足す


class TransformerMT(nn.Module):  # Transformer翻译模型 / Transformer翻訳モデル
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_size=256, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1):  # 初始化 / 初期化する
        super().__init__()  # 父类初始化 / 親クラス初期化
        self.src_embedding = nn.Embedding(src_vocab_size, emb_size, padding_idx=0)  # 源embedding / 入力embedding
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=0)  # 目标embedding / 目標embedding
        self.positional_encoding = PositionalEncoding(emb_size, dropout)  # 位置编码 / 位置符号化
        self.transformer = nn.Transformer(emb_size, nhead, num_layers, num_layers, dim_feedforward, dropout)  # Transformer主体 / Transformer本体
        self.generator = nn.Linear(emb_size, tgt_vocab_size)  # 输出层 / 出力層
        self.emb_size = emb_size  # 保存维度 / 次元を保存する

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):  # 前向计算 / 順伝播
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.emb_size))  # 源embedding / 入力embedding
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.emb_size))  # 目标embedding / 目標embedding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0), device=tgt.device)  # 未来mask / 未来mask
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)  # Transformer计算 / Transformer計算
        return self.generator(out)  # 返回logits / logitsを返す


def create_padding_mask(seq, pad_id=0):  # 创建padding mask / padding maskを作る
    return (seq == pad_id).transpose(0, 1)  # batch x seq_len / batch x seq_len


def train_epoch(model, loader, optimizer, loss_fn, device):  # 训练一轮 / 1epoch学習する
    model.train()  # 训练模式 / 学習モード
    total_loss = 0.0  # 累计loss / lossを累積する
    for src, tgt in tqdm(loader, desc="train", leave=False):  # 遍历batch / batchを走査する
        src, tgt = src.to(device), tgt.to(device)  # 移动设备 / デバイスへ移す
        tgt_input = tgt[:-1, :]  # decoder输入 / decoder入力
        tgt_out = tgt[1:, :]  # 目标输出 / 目標出力
        optimizer.zero_grad()  # 清空梯度 / 勾配を消す
        logits = model(src, tgt_input, create_padding_mask(src), create_padding_mask(tgt_input), create_padding_mask(src))  # 前向计算 / 順伝播
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))  # 计算loss / lossを計算する
        loss.backward()  # 反向传播 / 逆伝播
        optimizer.step()  # 更新参数 / パラメータ更新
        total_loss += loss.item()  # 累加loss / lossを加算する
    return total_loss / max(len(loader), 1)  # 平均loss / 平均loss


def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=80, device=None):  # greedy翻译 / greedy翻訳
    device = get_device() if device is None else device  # 自动设备 / 自動デバイス
    model.eval()  # 评价模式 / 評価モード
    ys = torch.tensor([[tgt_vocab[BOS]]], dtype=torch.long, device=device)  # 初始decoder输入 / 初期decoder入力
    src = src.to(device)  # 源移动设备 / 入力をデバイスへ移す
    with torch.no_grad():  # 不计算梯度 / 勾配なし
        for _ in range(max_len):  # 循环生成 / 繰り返し生成
            logits = model(src, ys, create_padding_mask(src), create_padding_mask(ys), create_padding_mask(src))  # 前向 / 順伝播
            next_id = logits[-1, 0].argmax().item()  # 取最大概率ID / 最大確率IDを取る
            ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=0)  # 追加token / tokenを追加する
            if next_id == tgt_vocab[EOS]:  # EOS停止 / EOSで停止
                break  # 结束 / 終了
    return decode(ys.squeeze(1).tolist(), tgt_vocab)  # 返回token列 / token列を返す


def save_checkpoint(path, model, src_vocab, tgt_vocab, config):  # 保存checkpoint / checkpointを保存する
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # 创建目录 / ディレクトリを作る
    torch.save({"model_state": model.state_dict(), "src_vocab": src_vocab, "tgt_vocab": tgt_vocab, "config": config}, path)  # 保存 / 保存する


def load_checkpoint(path, device=None):  # 读取checkpoint / checkpointを読む
    device = get_device() if device is None else device  # 自动设备 / 自動デバイス
    checkpoint = torch.load(path, map_location=device)  # 读取文件 / ファイルを読む
    config = checkpoint["config"]  # 取得配置 / 設定を取得する
    model = TransformerMT(len(checkpoint["src_vocab"]), len(checkpoint["tgt_vocab"]), **config).to(device)  # 创建模型 / モデルを作る
    model.load_state_dict(checkpoint["model_state"])  # 读取权重 / 重みを読む
    return model, checkpoint["src_vocab"], checkpoint["tgt_vocab"], config  # 返回对象 / オブジェクトを返す


def corpus_bleu(predictions, references):  # 计算BLEU / BLEUを計算する
    bleu = sacrebleu.corpus_bleu(predictions, [references])  # sacreBLEU计算 / sacreBLEU計算
    return bleu.score  # 返回分数 / スコアを返す
