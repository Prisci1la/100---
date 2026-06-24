'''
chapter8_utils.py: 第8章共用工具 / 第8章の共通ユーティリティ

集中管理词向量读取、SST-2数据集和共用参数解析。
/ 単語ベクトル読み込み、SST-2データセット、共通引数解析をまとめる。
'''

from __future__ import annotations  # 延迟类型注解的求值 / 型注釈の評価を遅延する

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ
import csv  # CSV和TSV读取库 / CSVとTSVを読むライブラリ
import re  # 正则表达式库 / 正規表現ライブラリ
from pathlib import Path  # 路径处理类 / パス処理クラス

import numpy as np  # 数值计算库 / 数値計算ライブラリ
import torch  # PyTorch主体库 / PyTorch本体ライブラリ
from torch.utils.data import Dataset  # 自定义数据集基类 / カスタムデータセット基底クラス


BASE_DIR = Path(__file__).resolve().parent  # 当前第8章目录 / 現在の第8章ディレクトリ
DEFAULT_TRAIN_PATH = BASE_DIR / "data" / "SST-2" / "train.tsv"  # 默认训练数据 / 既定の訓練データ
DEFAULT_DEV_PATH = BASE_DIR / "data" / "SST-2" / "dev.tsv"  # 默认开发数据 / 既定の開発データ
DEFAULT_VECTOR_PATH = BASE_DIR.parent / "Chapter 6" / "GoogleNews-vectors-negative300.bin"  # 默认词向量路径 / 既定の単語ベクトルパス


def build_embedding_resources(vector_path=DEFAULT_VECTOR_PATH, binary=True, max_vocab=None):  # 构建embedding矩阵和词表 / embedding行列と語彙を作る
    vectors = load_keyed_vectors(vector_path, binary=binary, limit=max_vocab)  # 读取预训练词向量 / 学習済み単語ベクトルを読む
    dim = vectors.vector_size  # 获取向量维度 / ベクトル次元数を取得する
    embedding_matrix = np.zeros((len(vectors.index_to_key) + 1, dim), dtype=np.float32)  # 第0行预留给PAD / 0行目をPAD用に空ける
    token_to_id = {"<PAD>": 0}  # 初始化token到ID的字典 / tokenからIDへの辞書を初期化する
    id_to_token = ["<PAD>"]  # 初始化ID到token的列表 / IDからtokenへの一覧を初期化する

    for row_id, token in enumerate(vectors.index_to_key, start=1):  # 从第1行开始填入词向量 / 1行目から単語ベクトルを入れる
        embedding_matrix[row_id] = vectors[token]  # 保存该token的向量 / そのtokenのベクトルを保存する
        token_to_id[token] = row_id  # 保存token到ID的映射 / tokenからIDへの対応を保存する
        id_to_token.append(token)  # 保存ID到token的映射 / IDからtokenへの対応を保存する

    return embedding_matrix, token_to_id, id_to_token  # 返回三个资源 / 3つのリソースを返す


class SSTDataset(Dataset):  # 自定义SST数据集类 / SSTのカスタムデータセットクラス
    def __init__(self, path, token_to_id):  # 初始化数据集 / データセットを初期化する
        self.examples = []  # 初始化有效样本列表 / 有効サンプル一覧を初期化する
        for row in load_sst_rows(path):  # 读取并遍历原始样本 / 元サンプルを読んで走査する
            input_ids = [token_to_id[token] for token in tokenize(row["text"]) if token in token_to_id]  # 转换成词ID / 単語IDへ変換する
            if not input_ids:  # 如果所有词都不在词表中 / 全単語が語彙外の場合
                continue  # 删除空token列样本 / 空token列の事例を除く
            self.examples.append({  # 追加一个样本字典 / サンプル辞書を追加する
                "text": row["text"],  # 保存原始文本 / 元テキストを保存する
                "label": torch.tensor([float(row["label"])], dtype=torch.float32),  # 保存浮点标签 / 浮動小数ラベルを保存する
                "input_ids": torch.tensor(input_ids, dtype=torch.long),  # 保存token ID列 / token ID列を保存する
            })

    def __len__(self):  # 返回样本数 / サンプル数を返す
        return len(self.examples)  # 返回有效样本长度 / 有効サンプル数を返す

    def __getitem__(self, index):  # 按索引返回样本 / 添字でサンプルを返す
        return self.examples[index]  # 返回对应样本 / 対応するサンプルを返す


def parse_common_args(description):  # 解析通用命令行参数 / 共通コマンドライン引数を解析する
    parser = argparse.ArgumentParser(description=description)  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--vector-path", default=str(DEFAULT_VECTOR_PATH), help="word2vec binary/text vector path")  # 词向量路径 / 単語ベクトルパス
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="SST-2 train csv/tsv path")  # 训练数据路径 / 訓練データパス
    parser.add_argument("--dev-path", default=str(DEFAULT_DEV_PATH), help="SST-2 dev csv/tsv path")  # 开发数据路径 / 開発データパス
    parser.add_argument("--max-vocab", type=int, default=50000, help="maximum vectors to load; 0 means all")  # 最大词表数量 / 最大語彙数
    parser.add_argument("--text-vectors", action="store_true", help="read vectors as text format")  # 文本格式词向量开关 / text形式ベクトルの指定
    return parser.parse_args()  # 解析并返回参数 / 解析して返す


def load_keyed_vectors(vector_path=DEFAULT_VECTOR_PATH, binary=True, limit=None):  # 读取Google News词向量 / Google News単語ベクトルを読む
    try:  # 尝试导入gensim / gensimの読み込みを試す
        from gensim.models import KeyedVectors  # 导入KeyedVectors / KeyedVectorsを導入する
    except ImportError as exc:  # gensim未安装时处理 / gensimが無い場合の処理
        raise RuntimeError("gensim is required. Run: python -m pip install gensim") from exc  # 提示安装依赖 / 依存関係の導入を案内する

    vector_path = Path(vector_path)  # 转成Path对象 / Pathオブジェクトに変換する
    if not vector_path.exists():  # 检查词向量文件是否存在 / 単語ベクトルファイルの存在確認
        raise FileNotFoundError(f"Word vector file was not found: {vector_path}")  # 文件不存在时报错 / 無ければエラーにする

    limit = None if limit is not None and limit <= 0 else limit  # 0以下なら全件読み込みにする / 0以下なら全件読み込みにする
    return KeyedVectors.load_word2vec_format(str(vector_path), binary=binary, limit=limit)  # 读取word2vec格式 / word2vec形式で読む


def load_sst_rows(path):  # 读取SST-2的CSV或TSV数据 / SST-2のCSVまたはTSVデータを読む
    path = Path(path)  # 转成Path对象 / Pathオブジェクトに変換する
    delimiter = "\t" if path.suffix == ".tsv" else ","  # 按扩展名选择分隔符 / 拡張子で区切り文字を選ぶ
    rows = []  # 初始化样本列表 / サンプル一覧を初期化する
    with path.open(encoding="utf-8", newline="") as file:  # 用UTF-8打开文件 / UTF-8でファイルを開く
        reader = csv.DictReader(file, delimiter=delimiter)  # 以表头作为字段名读取 / ヘッダーを列名として読む
        for row in reader:  # 逐行处理数据 / 各行を順に処理する
            text = row.get("sentence") or row.get("text")  # 读取文本列 / テキスト列を読む
            label = row.get("label")  # 读取标签列 / ラベル列を読む
            if text is None or label is None:  # 如果缺少必要列就跳过 / 必須列が無ければ読み飛ばす
                continue  # 进入下一行 / 次の行へ進む
            rows.append({"text": text.strip(), "label": int(label)})  # 保存文本和整数标签 / テキストと整数ラベルを保存する
    return rows  # 返回样本列表 / サンプル一覧を返す


def tokenize(text):  # 将英文文本切成简单token / 英文テキストを簡単なtokenに分ける
    return re.findall(r"[A-Za-z0-9_']+|[^\sA-Za-z0-9_']", text)  # 保留单词和标点 / 単語と記号を残す
