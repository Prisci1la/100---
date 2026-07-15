'''
knock90.py: データの準備 / データの準備

KFTTをダウンロードし、train/dev/testを形態素・単語単位で前処理する。
/ KFTTをダウンロードし、train/dev/testを形態素・単語単位で前処理する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from chapter13_utils import PROCESSED_DIR, build_vocab, download_kftt, extract_kftt, load_token_lines, preprocess_split, save_vocab  # 共用工具 / 共通ツール


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

