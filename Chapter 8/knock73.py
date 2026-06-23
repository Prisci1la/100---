'''
knock73.py: 模型的学习 / モデルの学習

在训练集上学习问题72的BoW模型，并固定词嵌入矩阵。
/ 問題72のBoWモデルを訓練セットで学習し、単語埋め込み行列は固定する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # 导入PyTorch / PyTorchを導入する
from torch.utils.data import DataLoader  # 导入DataLoader / DataLoaderを導入する

from chapter8_utils import BoWClassifier, SSTDataset, add_common_arguments, build_embedding_resources, collate_bow, get_device  # 导入第8章工具 / 第8章ツールを導入する


def evaluate(model, dataloader, device):  # 定义评估函数 / 評価関数を定義する
    model.eval()  # 切换到评估模式 / 評価モードに切り替える
    correct = 0  # 初始化正确数 / 正解数を初期化する
    total = 0  # 初始化总数 / 総数を初期化する
    with torch.no_grad():  # 评估时不计算梯度 / 評価時は勾配を計算しない
        for batch in dataloader:  # 遍历开发集batch / 開発セットbatchを走査する
            input_ids = batch["input_ids"].to(device)  # 将输入移动到设备 / 入力をデバイスへ移す
            lengths = batch["lengths"].to(device)  # 将长度移动到设备 / 長さをデバイスへ移す
            labels = batch["label"].to(device)  # 将标签移动到设备 / ラベルをデバイスへ移す
            logits = model(input_ids, lengths)  # 计算logit / logitを計算する
            predictions = (torch.sigmoid(logits) >= 0.5).float()  # 转换为0/1预测 / 0/1予測へ変換する
            correct += (predictions == labels).sum().item()  # 累加正确数 / 正解数を加算する
            total += labels.numel()  # 累加样本数 / サンプル数を加算する
    return correct / total if total else 0.0  # 返回准确率 / 正解率を返す


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):  # 定义一轮训练 / 1epochの学習を定義する
    model.train()  # 切换到训练模式 / 学習モードに切り替える
    total_loss = 0.0  # 初始化损失总和 / 損失合計を初期化する
    for batch in dataloader:  # 遍历训练batch / 訓練batchを走査する
        input_ids = batch["input_ids"].to(device)  # 将输入移动到设备 / 入力をデバイスへ移す
        lengths = batch["lengths"].to(device)  # 将长度移动到设备 / 長さをデバイスへ移す
        labels = batch["label"].to(device)  # 将标签移动到设备 / ラベルをデバイスへ移す
        optimizer.zero_grad()  # 清空上一轮梯度 / 前回の勾配を消す
        logits = model(input_ids, lengths)  # 前向计算 / 順伝播を行う
        loss = loss_fn(logits, labels)  # 计算二分类损失 / 二値分類損失を計算する
        loss.backward()  # 反向传播 / 逆伝播を行う
        optimizer.step()  # 更新分类层参数 / 分類層パラメータを更新する
        total_loss += loss.item() * labels.size(0)  # 累加batch损失 / batch損失を加算する
    return total_loss / len(dataloader.dataset)  # 返回平均损失 / 平均損失を返す


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock73: train BoW classifier")  # 创建参数解析器 / 引数パーサーを作る
    add_common_arguments(parser)  # 添加第8章通用参数 / 第8章共通引数を追加する
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")  # 训练轮数 / 学習epoch数
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")  # batch大小 / batchサイズ
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")  # 学习率 / 学習率
    args = parser.parse_args()  # 解析命令行参数 / コマンドライン引数を解析する

    device = get_device()  # 获取CPU或GPU设备 / CPUまたはGPUデバイスを取得する
    print(f"Using device: {device}")  # 输出当前设备 / 現在のデバイスを出力する
    if device.type == "cuda":  # 如果使用CUDA / CUDAを使う場合
        print(f"GPU: {torch.cuda.get_device_name(0)}")  # 输出GPU名称 / GPU名を出力する

    embedding_matrix, token_to_id, _id_to_token = build_embedding_resources(  # 构建embedding资源 / embeddingリソースを構築する
        vector_path=args.vector_path,  # 指定词向量路径 / 単語ベクトルパスを指定する
        binary=not args.text_vectors,  # 指定词向量格式 / 単語ベクトル形式を指定する
        max_vocab=args.max_vocab,  # 限制词表大小 / 語彙数を制限する
    )
    train_dataset = SSTDataset(args.train_path, token_to_id)  # 创建训练Dataset / 訓練Datasetを作る
    dev_dataset = SSTDataset(args.dev_path, token_to_id)  # 创建开发Dataset / 開発Datasetを作る
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_bow)  # 创建训练DataLoader / 訓練DataLoaderを作る
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bow)  # 创建开发DataLoader / 開発DataLoaderを作る

    model = BoWClassifier(embedding_matrix, freeze_embeddings=True).to(device)  # 创建模型并移动到设备 / モデルを作ってデバイスへ移す
    loss_fn = torch.nn.BCEWithLogitsLoss()  # 创建二分类损失函数 / 二値分類損失関数を作る
    optimizer = torch.optim.Adam(model.linear.parameters(), lr=args.lr)  # 只优化分类层参数 / 分類層だけを最適化する

    for epoch in range(1, args.epochs + 1):  # 按epoch循环 / epochごとに繰り返す
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)  # 训练一轮 / 1epoch学習する
        dev_accuracy = evaluate(model, dev_loader, device)  # 计算开发集准确率 / 開発セット正解率を計算する
        print(f"epoch {epoch:02d}: loss={train_loss:.6f}, dev_accuracy={dev_accuracy:.6f}")  # 输出训练进度 / 学習進捗を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

r'''
运行结果: / 実行結果:
Using device: cuda
GPU: NVIDIA GeForce RTX 5070 Ti
epoch 01: loss=0.648294, dev_accuracy=0.606651
epoch 02: loss=0.582137, dev_accuracy=0.709862
epoch 03: loss=0.541365, dev_accuracy=0.730505
epoch 04: loss=0.513981, dev_accuracy=0.746560
epoch 05: loss=0.494448, dev_accuracy=0.746560
'''
