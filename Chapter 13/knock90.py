'''
knock90.py: データの準備 / データの準備

KFTTをダウンロードし、train/dev/testを形態素・単語単位で前処理する。
/ KFTTをダウンロードし、train/dev/testを形態素・単語単位で前処理する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ
import json  # JSON保存库 / JSON保存ライブラリ
import tarfile  # tar.gz展开库 / tar.gz展開ライブラリ
import urllib.request  # 下载库 / ダウンロードライブラリ
from collections import Counter  # 词频统计 / 語彙頻度集計
from pathlib import Path  # 路径处理 / パス処理

from tqdm.auto import tqdm  # 进度条 / 進捗バー


BASE_DIR = Path(__file__).resolve().parent  # 当前第13回目录 / 現在の第13回ディレクトリ
KFTT_URL = "https://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz"  # KFTT下载地址 / KFTTダウンロード先
RAW_DIR = BASE_DIR / "data" / "raw"  # 原始数据目录 / 生データディレクトリ
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # 前处理数据目录 / 前処理データディレクトリ
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]  # 特殊token列表 / 特殊token一覧


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
        raise FileNotFoundError(f"Could not find KFTT {split} files under {kftt_dir}")  # 明确报告缺失文件 / 不足ファイルを明示する
    return ja_candidates[0], en_candidates[0]  # 返回第一组候选 / 最初の候補を返す


def get_ja_tagger():  # 创建日语分词器 / 日本語分かち書き器を作る
    try:  # 优先使用fugashi / fugashiを優先する
        import fugashi  # 导入日语形态素分析器 / 日本語形態素解析器を導入する
        return fugashi.Tagger()  # 返回分词器 / 分かち書き器を返す
    except Exception as exc:  # 失败时明确报错 / 失敗時は明確にエラー
        raise RuntimeError("Japanese morphological tokenizer is required. Install fugashi and unidic-lite.") from exc  # 补充安装方法后重新抛错 / 導入方法を添えて再送出する


def tokenize_ja(text, tagger=None):  # 日语分词 / 日本語を分かち書きする
    if tagger is None:  # 未传入分词器时 / 分かち書き器が未指定の場合
        tagger = get_ja_tagger()  # 自动创建分词器 / 分かち書き器を自動作成する
    return [word.surface for word in tagger(text.strip())]  # fugashi分词 / fugashiで分かち書きする


def tokenize_en(text):  # 英语tokenize / 英語tokenize
    return text.strip().lower().split()  # 简单按空格切分 / 簡単に空白で分ける


def preprocess_split(kftt_dir, split, out_dir=PROCESSED_DIR, max_lines=None):  # 前处理一个split / 1つのsplitを前処理する
    out_dir = Path(out_dir)  # 转换为Path / Pathへ変換する
    out_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录 / 出力ディレクトリを作る
    ja_path, en_path = find_parallel_files(kftt_dir, split)  # 查找日英文件 / 日英ファイルを探す
    tagger = get_ja_tagger()  # 创建日语分词器 / 日本語分かち書き器を作る
    out_ja = out_dir / f"{split}.ja.tok"  # 日语输出路径 / 日本語出力パス
    out_en = out_dir / f"{split}.en.tok"  # 英语输出路径 / 英語出力パス
    count = 0  # 初始化有效句对数 / 有効な対訳数を初期化する
    with ja_path.open(encoding="utf-8") as ja_file, en_path.open(encoding="utf-8") as en_file, out_ja.open("w", encoding="utf-8") as ja_out, out_en.open("w", encoding="utf-8") as en_out:  # 同时打开输入输出文件 / 入出力ファイルを同時に開く
        for ja_line, en_line in tqdm(zip(ja_file, en_file), desc=f"preprocess {split}", leave=False):  # 逐行读取平行语料 / 対訳コーパスを1行ずつ読む
            ja_tokens = tokenize_ja(ja_line, tagger)  # 日语分词 / 日本語を分かち書きする
            en_tokens = tokenize_en(en_line)  # 英语分词 / 英語を分かち書きする
            if not ja_tokens or not en_tokens:  # 跳过空句对 / 空の対訳を飛ばす
                continue  # 处理下一句 / 次の文を処理する
            ja_out.write(" ".join(ja_tokens) + "\n")  # 写入日语token / 日本語tokenを書き込む
            en_out.write(" ".join(en_tokens) + "\n")  # 写入英语token / 英語tokenを書き込む
            count += 1  # 增加有效句对数 / 有効な対訳数を増やす
            if max_lines is not None and count >= max_lines:  # 达到指定上限 / 指定上限に達した場合
                break  # 结束预处理 / 前処理を終了する
    return out_ja, out_en, count  # 返回输出路径和句对数 / 出力パスと対訳数を返す


def load_token_lines(path):  # 读取tokenized文件 / tokenizedファイルを読む
    return [line.strip().split() for line in Path(path).open(encoding="utf-8") if line.strip()]  # 返回非空token列 / 空でないtoken列を返す


def build_vocab(token_lines, max_size=50000, min_freq=2):  # 构建词表 / 語彙を構築する
    counter = Counter(token for line in token_lines for token in line)  # 统计全部token频率 / 全tokenの頻度を数える
    vocab = {token: index for index, token in enumerate(SPECIALS)}  # 先登记特殊token / 特殊tokenを先に登録する
    for token, freq in counter.most_common(max_size):  # 按频率从高到低遍历 / 頻度の高い順に走査する
        if freq < min_freq or token in vocab:  # 排除低频词和重复词 / 低頻語と重複語を除く
            continue  # 检查下一个token / 次のtokenを調べる
        vocab[token] = len(vocab)  # 分配连续ID / 連続したIDを割り当てる
    return vocab  # 返回构建好的词表 / 構築した語彙を返す


def save_vocab(vocab, path):  # 保存词表 / 語彙を保存する
    Path(path).write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")  # 以UTF-8 JSON保存 / UTF-8のJSONで保存する


def main():  # 主函数 / メイン関数
    parser = argparse.ArgumentParser(description="knock90: prepare KFTT data")  # 参数解析 / 引数解析
    parser.add_argument("--max-lines", type=int, default=None, help="limit lines for quick preparation")  # 行数上限 / 行数上限
    parser.add_argument("--vocab-size", type=int, default=50000, help="maximum vocabulary size")  # 词表大小 / 語彙サイズ
    parser.add_argument("--min-freq", type=int, default=2, help="minimum frequency")  # 最低频率 / 最低頻度
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    archive = download_kftt()  # 下载KFTT / KFTTをダウンロードする
    kftt_dir = extract_kftt(archive)  # 展开KFTT / KFTTを展開する
    counts = {}  # 保存行数 / 行数を保存する
    for split in ["train", "dev", "test"]:  # 前处理各split / 各splitを前処理する
        _ja, _en, count = preprocess_split(kftt_dir, split, PROCESSED_DIR, args.max_lines)  # 前处理 / 前処理
        counts[split] = count  # 保存行数 / 行数を保存する
    src_vocab = build_vocab(load_token_lines(PROCESSED_DIR / "train.ja.tok"), args.vocab_size, args.min_freq)  # 构建日语词表 / 日本語語彙を作る
    tgt_vocab = build_vocab(load_token_lines(PROCESSED_DIR / "train.en.tok"), args.vocab_size, args.min_freq)  # 构建英语词表 / 英語語彙を作る
    save_vocab(src_vocab, PROCESSED_DIR / "vocab.ja.json")  # 保存日语词表 / 日本語語彙を保存する
    save_vocab(tgt_vocab, PROCESSED_DIR / "vocab.en.json")  # 保存英语词表 / 英語語彙を保存する
    print("=" * 50)  # 分隔线 / 区切り線
    print("Knock 90: KFTT Preparation")  # 标题 / タイトル
    print("=" * 50)  # 分隔线 / 区切り線
    print(f"line counts: {counts}")  # 输出行数 / 行数を出力する
    print(f"ja vocab: {len(src_vocab)}")  # 输出日语词表大小 / 日本語語彙サイズ
    print(f"en vocab: {len(tgt_vocab)}")  # 输出英语词表大小 / 英語語彙サイズ


if __name__ == "__main__":  # 直接运行 / 直接実行
    main()  # 调用主函数 / メイン関数を呼ぶ


# AIX実行結果メモ (2026-07-18, log: ~/100knock/logs/ch13_knock90.log)
# KFTT preprocessing completed and wrote raw/processed data under Chapter 13/data/.
# line counts: {'train': 440286, 'dev': 1166, 'test': 1160}
# ja vocab: 50004; en vocab: 50004
# files produced include train/dev/test tokenized ja/en files and vocab.ja.json/vocab.en.json.
