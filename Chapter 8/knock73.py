'''
knock73.py: 模型的学习 / モデルの学習

在训练集上学习问题72的BoW模型，并固定词嵌入矩阵。
/ 問題72のBoWモデルを訓練セットで学習し、単語埋め込み行列は固定する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # 导入PyTorch / PyTorchを導入する
from torch import nn  # 导入神经网络模块 / ニューラルネットワークモジュールを導入する

from chapter8_utils import DEFAULT_TRAIN_PATH, DEFAULT_VECTOR_PATH, SSTDataset, build_embedding_resources  # 导入第8章工具 / 第8章ツールを導入する


def get_device():  # 获取训练设备 / 学習デバイスを取得する
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有GPU就使用CUDA / GPUがあればCUDAを使う


def add_common_arguments(parser):  # 添加第8章通用命令行参数 / 第8章共通コマンドライン引数を追加する
    parser.add_argument("--vector-path", default=str(DEFAULT_VECTOR_PATH), help="word2vec binary/text vector path")  # 词向量路径 / 単語ベクトルパス
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="SST-2 train csv/tsv path")  # 训练数据路径 / 訓練データパス
    parser.add_argument("--max-vocab", type=int, default=50000, help="maximum vectors to load; 0 means all")  # 最大词表数量 / 最大語彙数
    parser.add_argument("--text-vectors", action="store_true", help="read vectors as text format")  # 文本格式词向量开关 / text形式ベクトルの指定
    return parser  # 返回parser / parserを返す


class BoWClassifier(nn.Module):  # BoW二分类模型 / BoW二値分類モデル
    def __init__(self, embedding_matrix, freeze_embeddings=True):  # 初始化模型 / モデルを初期化する
        super().__init__()  # 调用父类初始化 / 親クラスを初期化する
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)  # 将numpy矩阵转为Tensor / numpy行列をTensorへ変換する
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=freeze_embeddings, padding_idx=0)  # 创建embedding层 / embedding層を作る
        self.linear = nn.Linear(embedding_tensor.shape[1], 1)  # 创建线性分类层 / 線形分類層を作る

    def forward(self, input_ids, lengths=None):  # 前向计算 / 順伝播を行う
        embedded = self.embedding(input_ids)  # 将token ID转换为词向量 / token IDを単語ベクトルへ変換する
        mask = (input_ids != 0).unsqueeze(-1)  # 创建PAD以外的掩码 / PAD以外のマスクを作る
        summed = (embedded * mask).sum(dim=1)  # 对有效词向量求和 / 有効な単語ベクトルを合計する
        if lengths is None:  # 如果没有传入长度 / 長さが渡されていない場合
            lengths = mask.sum(dim=1).clamp(min=1).squeeze(-1)  # 从掩码计算长度 / マスクから長さを計算する
        averaged = summed / lengths.to(embedded.device).unsqueeze(-1).clamp(min=1)  # 计算平均词向量 / 平均単語ベクトルを計算する
        return self.linear(averaged)  # 输出logit / logitを出力する


def train_one_epoch(model, dataset, optimizer, loss_fn, device):  # 定义一轮训练 / 1epochの学習を定義する
    model.train()  # 切换到训练模式 / 学習モードに切り替える
    total_loss = 0.0  # 初始化损失总和 / 損失合計を初期化する
    for example in dataset:  # 逐条遍历训练样本 / 訓練サンプルを1件ずつ走査する
        input_ids = example["input_ids"].unsqueeze(0).to(device)  # 将输入移动到设备 / 入力をデバイスへ移す
        lengths = torch.tensor([input_ids.size(1)], dtype=torch.long, device=device)  # 保存真实长度 / 実際の長さを保存する
        label = example["label"].unsqueeze(0).to(device)  # 将标签移动到设备 / ラベルをデバイスへ移す
        optimizer.zero_grad()  # 清空上一轮梯度 / 前回の勾配を消す
        logits = model(input_ids, lengths)  # 前向计算 / 順伝播を行う
        loss = loss_fn(logits, label)  # 计算二分类损失 / 二値分類損失を計算する
        loss.backward()  # 反向传播 / 逆伝播を行う
        optimizer.step()  # 更新分类层参数 / 分類層パラメータを更新する
        total_loss += loss.item()  # 累加样本损失 / サンプル損失を加算する
    return total_loss / len(dataset)  # 返回平均损失 / 平均損失を返す


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock73: train BoW classifier")  # 创建参数解析器 / 引数パーサーを作る
    add_common_arguments(parser)  # 添加第8章通用参数 / 第8章共通引数を追加する
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")  # 训练轮数 / 学習epoch数
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")  # 学习率 / 学習率
    args = parser.parse_args()  # 解析命令行参数 / コマンドライン引数を解析する

    device = get_device()  # 获取CPU或GPU设备 / CPUまたはGPUデバイスを取得する
    print(f"Using device: {device}")  # 输出当前设备 / 現在のデバイスを出力する

    embedding_matrix, token_to_id, _id_to_token = build_embedding_resources(  # 构建embedding资源 / embeddingリソースを構築する
        vector_path=args.vector_path,  # 指定词向量路径 / 単語ベクトルパスを指定する
        binary=not args.text_vectors,  # 指定词向量格式 / 単語ベクトル形式を指定する
        max_vocab=args.max_vocab,  # 限制词表大小 / 語彙数を制限する
    )
    train_dataset = SSTDataset(args.train_path, token_to_id)  # 创建训练Dataset / 訓練Datasetを作る
    model = BoWClassifier(embedding_matrix, freeze_embeddings=True).to(device)  # 创建模型并移动到设备 / モデルを作ってデバイスへ移す
    loss_fn = torch.nn.BCEWithLogitsLoss()  # 创建二分类损失函数 / 二値分類損失関数を作る
    optimizer = torch.optim.Adam(model.linear.parameters(), lr=args.lr)  # 只优化分类层参数 / 分類層だけを最適化する

    for epoch in range(1, args.epochs + 1):  # 按epoch循环 / epochごとに繰り返す
        train_loss = train_one_epoch(model, train_dataset, optimizer, loss_fn, device)  # 训练一轮 / 1epoch学習する
        print(f"epoch {epoch:02d}: loss={train_loss:.6f}")  # 输出训练进度 / 学習進捗を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

r'''
运行结果: / 実行結果:
Using device: cuda
epoch 01: loss=0.433131
epoch 02: loss=0.403585
epoch 03: loss=0.401043
epoch 04: loss=0.400207
epoch 05: loss=0.399837
'''
