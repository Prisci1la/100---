'''
chapter14_utils.py: 第14回共用工具 / 第14回の共通ユーティリティ

ビーム探索、SentencePiece、BLEU評価、追加コーパス処理、Flask翻訳用処理をまとめる。
/ ビーム探索、SentencePiece、BLEU評価、追加コーパス処理、Flask翻訳用処理をまとめる。
'''

from __future__ import annotations  # 延迟类型注解的求值 / 型注釈の評価を遅延する

import json  # JSON保存库 / JSON保存ライブラリ
import shutil  # 文件复制工具 / ファイルコピー用
import sys  # import路径处理 / importパス処理
import tempfile  # 临时目录工具 / 一時ディレクトリ用
from pathlib import Path  # 路径处理 / パス処理

import torch  # PyTorch / PyTorch
from torch.nn import functional as F  # 函数式API / 関数型API
from tqdm.auto import tqdm  # 进度条 / 進捗バー


BASE_DIR = Path(__file__).resolve().parent  # 当前第14回目录 / 現在の第14回ディレクトリ
CHAPTER13_DIR = BASE_DIR.parent / "Chapter 13"  # Chapter 13目录 / Chapter 13ディレクトリ
if str(CHAPTER13_DIR) not in sys.path:  # 如果未加入import路径 / importパス未追加なら
    sys.path.append(str(CHAPTER13_DIR))  # 加入import路径 / importパスへ追加する

from chapter13_utils import (  # noqa: E402  # 导入第13回工具 / 第13回ツールを導入する
    BOS,
    CHECKPOINT_DIR as CH13_CHECKPOINT_DIR,
    EOS,
    OUTPUT_DIR as CH13_OUTPUT_DIR,
    PROCESSED_DIR as CH13_PROCESSED_DIR,
    TransformerMT,
    build_vocab,
    collate_translation,
    corpus_bleu,
    create_padding_mask,
    decode,
    encode,
    get_device,
    get_ja_tagger,
    greedy_decode,
    load_checkpoint,
    load_token_lines,
    load_vocab,
    save_checkpoint,
    save_vocab,
    tokenize_en,
    tokenize_ja,
    train_epoch,
    TranslationDataset,
)


OUTPUT_DIR = BASE_DIR / "outputs"  # 输出目录 / 出力ディレクトリ
DATA_DIR = BASE_DIR / "data"  # 数据目录 / データディレクトリ
SERVER_DIR = BASE_DIR / "server"  # server目录 / serverディレクトリ
CHECKPOINT_DIR = BASE_DIR / "checkpoints"  # 第14回输出模型目录 / 第14回の出力モデルディレクトリ
BASE_CHECKPOINT = CH13_CHECKPOINT_DIR / "transformer_mt.pt"  # 第13回训练好的基础模型 / 第13回で学習した基礎モデル


def beam_search_decode(model, src, src_vocab, tgt_vocab, beam_size=5, max_len=80, device=None):  # ビーム探索でdecode / ビーム探索でdecodeする
    device = get_device() if device is None else device  # 自动设备 / 自動デバイス
    model.eval()  # 评价模式 / 評価モード
    src = src.to(device)  # 输入移动设备 / 入力をデバイスへ移す
    bos_id = tgt_vocab[BOS]  # BOS ID / BOS ID
    eos_id = tgt_vocab[EOS]  # EOS ID / EOS ID
    beams = [(torch.tensor([[bos_id]], dtype=torch.long, device=device), 0.0)]  # 初始化beam / beamを初期化する
    finished = []  # 完成候选 / 完了候補
    with torch.no_grad():  # 不计算梯度 / 勾配なし
        for _ in range(max_len):  # 最多生成max_len步 / 最大max_lenステップ生成する
            new_beams = []  # 新beam列表 / 新beamリスト
            for ys, score in beams:  # 遍历当前beam / 現在のbeamを走査する
                if ys[-1, 0].item() == eos_id:  # 已生成EOS / EOS生成済み
                    finished.append((ys, score))  # 加入完成 / 完了へ追加
                    continue  # 跳过扩展 / 展開しない
                logits = model(src, ys, create_padding_mask(src), create_padding_mask(ys), create_padding_mask(src))  # 前向 / 順伝播
                log_probs = F.log_softmax(logits[-1, 0], dim=-1)  # 取得下一token log概率 / 次token log確率を取得する
                top_scores, top_ids = torch.topk(log_probs, beam_size)  # 取top-k / top-kを取る
                for token_score, token_id in zip(top_scores.tolist(), top_ids.tolist()):  # 扩展beam / beamを展開する
                    next_ys = torch.cat([ys, torch.tensor([[token_id]], device=device)], dim=0)  # 追加token / tokenを追加する
                    new_beams.append((next_ys, score + token_score))  # 保存新候选 / 新候補を保存する
            if not new_beams:  # 没有新候选 / 新候補がない場合
                break  # 停止 / 停止する
            beams = sorted(new_beams, key=lambda item: item[1] / max(item[0].size(0), 1), reverse=True)[:beam_size]  # 平均log概率排序 / 平均log確率で並べる
    finished.extend(beams)  # 加入未完成beam / 未完了beamも加える
    best = max(finished, key=lambda item: item[1] / max(item[0].size(0), 1))[0]  # 选最好候选 / 最良候補を選ぶ
    return decode(best.squeeze(1).tolist(), tgt_vocab)  # 返回token列 / token列を返す


def evaluate_bleu_for_beam(model, src_lines, ref_lines, src_vocab, tgt_vocab, beam_size=5, max_examples=None, device=None):  # 指定beamのBLEUを評価 / 指定beamのBLEUを評価する
    predictions = []  # 初始化预测 / 予測を初期化する
    selected_src = src_lines[:max_examples]  # 截取源句 / 入力文を切り出す
    selected_ref = ref_lines[:max_examples]  # 截取参考译文 / 参照訳を切り出す
    for tokens in tqdm(selected_src, desc=f"beam={beam_size}", leave=False):  # 遍历源句 / 入力文を走査する
        src_ids = torch.tensor(encode(tokens, src_vocab), dtype=torch.long).unsqueeze(1)  # 编码源句 / 入力文を符号化する
        pred = beam_search_decode(model, src_ids, src_vocab, tgt_vocab, beam_size=beam_size, device=device)  # beam decode / beam decodeする
        predictions.append(" ".join(pred))  # 保存预测 / 予測を保存する
    return corpus_bleu(predictions, selected_ref), predictions  # 返回BLEU和预测 / BLEUと予測を返す


def train_sentencepiece(input_path, model_prefix, vocab_size=8000, character_coverage=0.9995):  # 训练SentencePiece / SentencePieceを学習する
    import sentencepiece as spm  # 延迟导入 / 遅延import
    model_prefix = Path(model_prefix)  # 转换为Path / Pathへ変換する
    model_prefix.parent.mkdir(parents=True, exist_ok=True)  # 创建目录 / ディレクトリを作る
    target_model = model_prefix.with_suffix(".model")  # 目标model路径 / 出力modelパス
    target_vocab = model_prefix.with_suffix(".vocab")  # 目标vocab路径 / 出力vocabパス
    with tempfile.TemporaryDirectory(prefix="spm_") as temp_dir:  # 避免非ASCII路径问题 / 非ASCIIパス問題を避ける
        temp_dir = Path(temp_dir)  # 转换为Path / Pathへ変換する
        temp_input = temp_dir / "input.txt"  # ASCII临时输入 / ASCII一時入力
        temp_prefix = temp_dir / model_prefix.name  # ASCII临时prefix / ASCII一時prefix
        shutil.copyfile(input_path, temp_input)  # 复制语料 / コーパスをコピーする
        spm.SentencePieceTrainer.train(  # 训练SP模型 / SPモデルを学習する
            input=str(temp_input),
            model_prefix=str(temp_prefix),
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type="unigram",
            normalization_rule_name="identity",
            hard_vocab_limit=False,
        )
        shutil.copyfile(temp_prefix.with_suffix(".model"), target_model)  # 复制模型 / モデルをコピーする
        shutil.copyfile(temp_prefix.with_suffix(".vocab"), target_vocab)  # 复制词表 / 語彙をコピーする
    return target_model  # 返回模型路径 / モデルパスを返す


def encode_with_sentencepiece(model_path, input_path, output_path):  # 用SP模型编码文件 / SPモデルでファイルを符号化する
    import sentencepiece as spm  # 延迟导入 / 遅延import
    sp = spm.SentencePieceProcessor()  # 创建SP处理器 / SP処理器を作る
    sp.LoadFromSerializedProto(Path(model_path).read_bytes())  # 从bytes读取，避开非ASCII路径 / bytesから読み非ASCIIパスを避ける
    output_path = Path(output_path)  # Path转换 / Path変換
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建目录 / ディレクトリを作る
    with Path(input_path).open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as out:  # 打开文件 / ファイルを開く
        for line in src:  # 遍历行 / 行を走査する
            out.write(" ".join(sp.encode(line.strip(), out_type=str)) + "\n")  # 写入SP token / SP tokenを書き込む
    return output_path  # 返回输出路径 / 出力パスを返す


def write_json(path, obj):  # 保存JSON / JSONを保存する
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # 创建目录 / ディレクトリを作る
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")  # 写JSON / JSONを書く


def read_parallel_text(ja_path, en_path, max_lines=None):  # 读取追加平行语料 / 追加対訳コーパスを読む
    tagger = get_ja_tagger()  # 创建tagger / taggerを作る
    ja_lines, en_lines = [], []  # 初始化列表 / リストを初期化する
    with Path(ja_path).open(encoding="utf-8") as ja_file, Path(en_path).open(encoding="utf-8") as en_file:  # 打开文件 / ファイルを開く
        for ja, en in zip(ja_file, en_file):  # 遍历平行句 / 対訳文を走査する
            ja_lines.append(tokenize_ja(ja, tagger))  # 日语分词 / 日本語を分かち書きする
            en_lines.append(tokenize_en(en))  # 英语分词 / 英語を分かち書きする
            if max_lines is not None and len(ja_lines) >= max_lines:  # 上限检查 / 上限確認
                break  # 停止 / 停止する
    return ja_lines, en_lines  # 返回tokenized语料 / tokenizedコーパスを返す
