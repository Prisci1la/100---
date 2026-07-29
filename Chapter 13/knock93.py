'''
knock93.py: BLEUスコアの計測 / BLEUスコアの計測

91で学習したモデルを用いて、評価データ上のBLEUスコアをsacreBLEUで測定する。
/ 91で学習したモデルを用いて、評価データ上のBLEUスコアをsacreBLEUで測定する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ
import math  # 数学函数 / 数学関数
from pathlib import Path  # 路径处理 / パス処理

import sacrebleu  # BLEU计算库 / BLEU計算ライブラリ
import torch  # PyTorch / PyTorch
from torch import nn  # 神经网络模块 / ニューラルネットワーク
from tqdm.auto import tqdm  # 进度条 / 進捗バー


BASE_DIR = Path(__file__).resolve().parent  # 当前第13回目录 / 現在の第13回ディレクトリ
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # 前处理数据目录 / 前処理データディレクトリ
CHECKPOINT_DIR = BASE_DIR / "checkpoints"  # 模型保存目录 / モデル保存ディレクトリ
OUTPUT_DIR = BASE_DIR / "outputs"  # 输出目录 / 出力ディレクトリ
PAD = "<pad>"  # padding token / padding token
BOS = "<bos>"  # 开始token / 開始token
EOS = "<eos>"  # 结束token / 終了token
UNK = "<unk>"  # 未知token / 未知token


def get_device():  # 获取可用设备 / 利用可能なデバイスを取得する
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有GPU则用CUDA / GPUがあればCUDAを使う


def load_token_lines(path):  # 读取tokenized文件 / tokenizedファイルを読む
    return [line.strip().split() for line in Path(path).open(encoding="utf-8") if line.strip()]  # 返回非空token列 / 空でないtoken列を返す


def encode(tokens, vocab):  # token列转ID列 / token列をID列へ変換する
    return [vocab[BOS]] + [vocab.get(token, vocab[UNK]) for token in tokens] + [vocab[EOS]]  # 添加BOS/EOS并替换未知词 / BOS・EOSを付け未知語を置換する


def decode(ids, vocab):  # ID列转token列 / ID列をtoken列へ変換する
    inv_vocab = {idx: token for token, idx in vocab.items()}  # 创建ID到token的映射 / IDからtokenへの対応を作る
    tokens = []  # 初始化输出token / 出力tokenを初期化する
    for idx in ids:  # 按生成顺序遍历ID / 生成順にIDを走査する
        token = inv_vocab.get(int(idx), UNK)  # 未知ID转为UNK / 未知IDをUNKへ変換する
        if token == EOS:  # 到EOS时结束 / EOSに達した場合
            break  # 停止解码 / decodeを止める
        if token not in {BOS, PAD}:  # 排除BOS和padding / BOSとpaddingを除く
            tokens.append(token)  # 保存普通token / 通常tokenを保存する
    return tokens  # 返回解码结果 / decode結果を返す


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


def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=80, device=None):  # greedy翻译 / greedy翻訳
    device = get_device() if device is None else device  # 未指定时自动选择设备 / 未指定ならdeviceを自動選択する
    model.eval()  # 切换评估模式 / 評価modeへ切り替える
    ys = torch.tensor([[tgt_vocab[BOS]]], dtype=torch.long, device=device)  # 以BOS开始生成 / BOSから生成を始める
    src = src.to(device)  # 将输入移到推理设备 / 入力を推論deviceへ移す
    with torch.no_grad():  # 推理时禁用梯度 / 推論時は勾配を無効にする
        for _ in range(max_len):  # 最多生成max_len个token / 最大max_len tokenを生成する
            logits = model(src, ys, create_padding_mask(src), create_padding_mask(ys), create_padding_mask(src))  # 计算下一个token分布 / 次tokenの分布を計算する
            next_id = logits[-1, 0].argmax().item()  # 选择概率最大的ID / 確率最大のIDを選ぶ
            ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=0)  # 将ID追加到生成序列 / IDを生成列へ追加する
            if next_id == tgt_vocab[EOS]:  # 生成EOS时 / EOSを生成した場合
                break  # 提前结束 / 早期終了する
    return decode(ys.squeeze(1).tolist(), tgt_vocab)  # 将生成ID还原为token / 生成IDをtokenへ戻す


def load_checkpoint(path, device=None):  # 读取checkpoint / checkpointを読む
    device = get_device() if device is None else device  # 未指定时自动选择设备 / 未指定ならdeviceを自動選択する
    checkpoint = torch.load(path, map_location=device)  # 将checkpoint读取到指定设备 / checkpointを指定deviceへ読む
    config = checkpoint["config"]  # 读取模型配置 / model設定を読む
    model = TransformerMT(len(checkpoint["src_vocab"]), len(checkpoint["tgt_vocab"]), **config).to(device)  # 按保存配置重建模型 / 保存設定でmodelを再構築する
    model.load_state_dict(checkpoint["model_state"])  # 恢复训练权重 / 学習済み重みを復元する
    return model, checkpoint["src_vocab"], checkpoint["tgt_vocab"], config  # 返回模型、词表和配置 / model・語彙・設定を返す


def corpus_bleu(predictions, references):  # 计算BLEU / BLEUを計算する
    bleu = sacrebleu.corpus_bleu(predictions, [references])  # 计算corpus BLEU / corpus BLEUを計算する
    return bleu.score  # 返回BLEU分数 / BLEU scoreを返す


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


# AIX実行結果メモ (2026-07-18, log: ~/100knock/logs/ch13_knock93.log)
# test set size: 1160; translation progress reached 1160/1160 in about 03:32.
# test BLEU: 2.22
# predictions saved to Chapter 13/outputs/test_predictions.txt.
# BLEU is low, so this checkpoint should be treated as a baseline rather than a good MT model.
