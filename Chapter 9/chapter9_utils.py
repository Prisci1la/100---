'''
chapter9_utils.py: 第9章共用工具 / 第9章の共通ユーティリティ

集中管理词向量读取、SST-2数据集、padding、mini-batch训练和评价。
/ 単語ベクトル読み込み、SST-2データセット、padding、mini-batch学習、評価をまとめる。
'''

from __future__ import annotations  # 延迟类型注解的求值 / 型注釈の評価を遅延する

import csv  # CSV和TSV读取库 / CSVとTSVを読むライブラリ
import re  # 正则表达式库 / 正規表現ライブラリ
from pathlib import Path  # 路径处理类 / パス処理クラス

import numpy as np  # 数值计算库 / 数値計算ライブラリ
import torch  # PyTorch主体库 / PyTorch本体ライブラリ
from torch import nn  # 神经网络模块 / ニューラルネットワークモジュール
from torch.utils.data import DataLoader, Dataset  # 数据加载器和数据集基类 / DataLoaderとDataset基底クラス


BASE_DIR = Path(__file__).resolve().parent  # 当前第9章目录 / 現在の第9章ディレクトリ
CHAPTER8_DIR = BASE_DIR.parent / "Chapter 8"  # 复用第8章已经下载的数据 / 第8章で取得済みのデータを再利用する
DEFAULT_TRAIN_PATH = CHAPTER8_DIR / "data" / "SST-2" / "train.tsv"  # 默认训练数据 / 既定の訓練データ
DEFAULT_DEV_PATH = CHAPTER8_DIR / "data" / "SST-2" / "dev.tsv"  # 默认开发数据 / 既定の開発データ
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


def mean_pool_embeddings(embedded, input_ids):  # 计算PAD掩码下的平均词向量 / PADを除いた平均単語ベクトルを計算する
    mask = (input_ids != 0).unsqueeze(-1)  # 创建非PAD位置的掩码 / PAD以外の位置のマスクを作る
    summed = (embedded * mask).sum(dim=1)  # 只累加有效token的向量 / 有効tokenのベクトルだけを合計する
    lengths = mask.sum(dim=1).clamp(min=1)  # 计算每个句子的有效长度 / 各文の有効長を計算する
    return summed / lengths  # 返回平均后的文本表示 / 平均化したテキスト表現を返す


class BoWClassifier(nn.Module):  # BoW二分类模型 / BoW二値分類モデル
    def __init__(self, embedding_matrix, freeze_embeddings=True):  # 初始化模型 / モデルを初期化する
        super().__init__()  # 调用父类初始化 / 親クラスを初期化する
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)  # 将numpy矩阵转换为Tensor / numpy行列をTensorに変換する
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=freeze_embeddings, padding_idx=0)  # 构建预训练embedding层 / 学習済みembedding層を作る
        self.linear = nn.Linear(embedding_tensor.shape[1], 1)  # 构建线性分类层 / 線形分類層を作る

    def forward(self, input_ids):  # 前向计算 / 順伝播を行う
        embedded = self.embedding(input_ids)  # 将token ID转换为词向量 / token IDを単語ベクトルへ変換する
        averaged = mean_pool_embeddings(embedded, input_ids)  # 计算平均词向量 / 平均単語ベクトルを計算する
        return self.linear(averaged)  # 输出logit / logitを出力する


class MLPBoWClassifier(nn.Module):  # 多层BoW分类模型 / 多層BoW分類モデル
    def __init__(self, embedding_matrix, hidden_size=128, dropout=0.3, freeze_embeddings=True):  # 初始化MLP模型 / MLPモデルを初期化する
        super().__init__()  # 调用父类初始化 / 親クラスを初期化する
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)  # 将embedding矩阵转为Tensor / embedding行列をTensorに変換する
        emb_dim = embedding_tensor.shape[1]  # 取得词向量维度 / 単語ベクトル次元を取得する
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=freeze_embeddings, padding_idx=0)  # 创建embedding层 / embedding層を作る
        self.classifier = nn.Sequential(  # 创建多层分类器 / 多層分類器を作る
            nn.Linear(emb_dim, hidden_size),  # 第一层线性变换 / 1層目の線形変換
            nn.ReLU(),  # 非线性激活 / 非線形活性化
            nn.Dropout(dropout),  # Dropout正则化 / Dropout正則化
            nn.Linear(hidden_size, 1),  # 输出二分类logit / 二値分類logitを出力する
        )

    def forward(self, input_ids):  # 前向计算 / 順伝播を行う
        embedded = self.embedding(input_ids)  # 将token ID转换为词向量 / token IDを単語ベクトルへ変換する
        averaged = mean_pool_embeddings(embedded, input_ids)  # 计算文本平均向量 / テキスト平均ベクトルを計算する
        return self.classifier(averaged)  # 通过MLP输出logit / MLPでlogitを出力する


def get_device():  # 获取可用设备 / 利用可能なデバイスを取得する
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有GPU则使用CUDA / GPUがあればCUDAを使う


def collate_examples(examples):  # 将多个样本整理为一个batch / 複数サンプルを1つのbatchにまとめる
    if not examples:  # 空batch不能构造张量 / 空batchはテンソル化できない
        raise ValueError("collate_examples received an empty batch")  # 抛出明确错误 / 明確なエラーを出す

    ordered = sorted(examples, key=lambda example: len(example["input_ids"]), reverse=True)  # 按token长度降序排序 / token長の降順に並べる
    max_length = max(len(example["input_ids"]) for example in ordered)  # 找到最长token列 / 最長token列を求める
    input_ids = torch.zeros((len(ordered), max_length), dtype=torch.long)  # 用0初始化PAD后的矩阵 / 0でpadding済み行列を初期化する
    labels = torch.stack([example["label"] for example in ordered]).to(torch.float32)  # 堆叠标签 / ラベルを積み上げる

    for row_id, example in enumerate(ordered):  # 逐个样本写入矩阵 / サンプルごとに行列へ書き込む
        token_ids = example["input_ids"]  # 取得token ID列 / token ID列を取得する
        input_ids[row_id, : len(token_ids)] = token_ids  # 右侧不足部分保持为PAD 0 / 右側の不足分はPAD 0のままにする

    return {"input_ids": input_ids, "label": labels}  # 返回batch字典 / batch辞書を返す


def create_data_loader(dataset, batch_size=64, shuffle=False):  # 创建带collate的数据加载器 / collate付きDataLoaderを作る
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_examples)  # 返回DataLoader / DataLoaderを返す


def move_batch_to_device(batch, device):  # 将batch移动到设备 / batchをデバイスへ移動する
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}  # 只移动Tensor / Tensorだけ移動する


def train_one_epoch_batched(model, data_loader, optimizer, loss_fn, device):  # 训练一个mini-batch epoch / mini-batchで1epoch学習する
    model.train()  # 切换到训练模式 / 学習モードに切り替える
    total_loss = 0.0  # 累计损失 / 損失を累積する
    total_examples = 0  # 累计样本数 / サンプル数を累積する

    for batch in data_loader:  # 遍历mini-batch / mini-batchを走査する
        batch = move_batch_to_device(batch, device)  # 移动batch到设备 / batchをデバイスへ移す
        optimizer.zero_grad()  # 清空梯度 / 勾配を消す
        logits = model(batch["input_ids"])  # 前向计算 / 順伝播を行う
        loss = loss_fn(logits, batch["label"])  # 计算损失 / 損失を計算する
        loss.backward()  # 反向传播 / 逆伝播を行う
        optimizer.step()  # 更新参数 / パラメータを更新する

        batch_size = batch["label"].size(0)  # 当前batch大小 / 現在のbatchサイズ
        total_loss += loss.item() * batch_size  # 按样本数累计平均损失 / サンプル数で平均損失を累積する
        total_examples += batch_size  # 累计样本数 / サンプル数を足す

    return total_loss / max(total_examples, 1)  # 返回每个样本的平均损失 / サンプルあたり平均損失を返す


def evaluate_accuracy(model, data_loader, device):  # 计算数据集正解率 / データセット正解率を計算する
    model.eval()  # 切换到评价模式 / 評価モードに切り替える
    correct = 0  # 正确预测数 / 正解数
    total = 0  # 总预测数 / 全予測数

    with torch.no_grad():  # 评价时不计算梯度 / 評価時は勾配を計算しない
        for batch in data_loader:  # 遍历batch / batchを走査する
            batch = move_batch_to_device(batch, device)  # 移动到设备 / デバイスへ移す
            logits = model(batch["input_ids"])  # 前向计算 / 順伝播を行う
            predictions = (torch.sigmoid(logits) >= 0.5).to(batch["label"].dtype)  # 将概率转换为0/1 / 確率を0/1へ変換する
            correct += (predictions == batch["label"]).sum().item()  # 累加正确数量 / 正解数を加算する
            total += batch["label"].numel()  # 累加标签数量 / ラベル数を加算する

    return correct / max(total, 1)  # 返回正解率 / 正解率を返す


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
