'''
knock74.py: 正解率的计算 / 正解率の計測

训练问题73的BoW模型，并在开发集上计算正解率。
/ 問題73のBoWモデルを学習し、開発セットで正解率を計算する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # 导入PyTorch / PyTorchを導入する

from chapter9_utils import (  # 导入第9章共用工具 / 第9章共通ツールを導入する
    DEFAULT_DEV_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VECTOR_PATH,
    BoWClassifier,
    SSTDataset,
    build_embedding_resources,
    get_device,
)


def add_common_arguments(parser):  # 添加本题需要的命令行参数 / 本問で使うコマンドライン引数を追加する
    parser.add_argument("--vector-path", default=str(DEFAULT_VECTOR_PATH), help="word2vec binary/text vector path")  # 词向量路径 / 単語ベクトルパス
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="SST-2 train csv/tsv path")  # 训练数据路径 / 訓練データパス
    parser.add_argument("--dev-path", default=str(DEFAULT_DEV_PATH), help="SST-2 dev csv/tsv path")  # 开发数据路径 / 開発データパス
    parser.add_argument("--max-vocab", type=int, default=50000, help="maximum vectors to load; 0 means all")  # 最大词表数量 / 最大語彙数
    parser.add_argument("--text-vectors", action="store_true", help="read vectors as text format")  # 文本格式词向量开关 / text形式ベクトルの指定
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")  # 训练轮数 / 学習epoch数
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")  # 学习率 / 学習率
    return parser  # 返回parser / parserを返す


def train_one_epoch(model, dataset, optimizer, loss_fn, device):  # 训练一个epoch / 1epoch学習する
    model.train()  # 切换到训练模式 / 学習モードに切り替える
    total_loss = 0.0  # 累计损失 / 損失を累積する

    for example in dataset:  # 逐条处理样本 / サンプルを1件ずつ処理する
        input_ids = example["input_ids"].unsqueeze(0).to(device)  # 添加batch维度并移动到设备 / batch次元を足してデバイスへ移す
        label = example["label"].unsqueeze(0).to(device)  # 添加batch维度并移动标签 / ラベルにbatch次元を足して移す
        optimizer.zero_grad()  # 清空上一轮梯度 / 前回の勾配を消す
        logits = model(input_ids)  # 前向计算 / 順伝播を行う
        loss = loss_fn(logits, label)  # 计算二分类损失 / 二値分類損失を計算する
        loss.backward()  # 反向传播 / 逆伝播を行う
        optimizer.step()  # 更新分类层参数 / 分類層パラメータを更新する
        total_loss += loss.item()  # 累加损失 / 損失を加算する

    return total_loss / len(dataset)  # 返回平均损失 / 平均損失を返す


def evaluate_accuracy_single(model, dataset, device):  # 单样本方式计算正解率 / 1件ずつ正解率を計算する
    model.eval()  # 切换到评价模式 / 評価モードに切り替える
    correct = 0  # 正确预测数 / 正解数
    total = 0  # 总样本数 / 全サンプル数

    with torch.no_grad():  # 评价时不需要梯度 / 評価時は勾配を使わない
        for example in dataset:  # 遍历开发集 / 開発セットを走査する
            input_ids = example["input_ids"].unsqueeze(0).to(device)  # 添加batch维度 / batch次元を足す
            label = example["label"].unsqueeze(0).to(device)  # 标签也添加batch维度 / ラベルにもbatch次元を足す
            logits = model(input_ids)  # 得到logit / logitを得る
            prediction = (torch.sigmoid(logits) >= 0.5).to(label.dtype)  # 概率转为0或1 / 確率を0または1へ変換する
            correct += (prediction == label).sum().item()  # 累加正确数 / 正解数を加算する
            total += label.numel()  # 累加样本数 / サンプル数を加算する

    return correct / max(total, 1)  # 返回正解率 / 正解率を返す


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock74: evaluate BoW classifier accuracy")  # 创建参数解析器 / 引数パーサーを作る
    add_common_arguments(parser)  # 添加参数 / 引数を追加する
    args = parser.parse_args()  # 解析命令行参数 / コマンドライン引数を解析する

    device = get_device()  # 获取CPU或GPU设备 / CPUまたはGPUデバイスを取得する
    embedding_matrix, token_to_id, _id_to_token = build_embedding_resources(  # 构建embedding资源 / embeddingリソースを構築する
        vector_path=args.vector_path,  # 指定词向量路径 / 単語ベクトルパスを指定する
        binary=not args.text_vectors,  # 指定词向量格式 / 単語ベクトル形式を指定する
        max_vocab=args.max_vocab,  # 限制词表大小 / 語彙数を制限する
    )
    train_dataset = SSTDataset(args.train_path, token_to_id)  # 创建训练Dataset / 訓練Datasetを作る
    dev_dataset = SSTDataset(args.dev_path, token_to_id)  # 创建开发Dataset / 開発Datasetを作る
    model = BoWClassifier(embedding_matrix, freeze_embeddings=True).to(device)  # 创建BoW模型 / BoWモデルを作る
    loss_fn = torch.nn.BCEWithLogitsLoss()  # 创建损失函数 / 損失関数を作る
    optimizer = torch.optim.Adam(model.linear.parameters(), lr=args.lr)  # 只训练线性分类层 / 線形分類層だけを学習する

    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 74: Accuracy")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(f"device: {device}")  # 输出设备 / デバイスを出力する

    for epoch in range(1, args.epochs + 1):  # 按epoch循环 / epochごとに繰り返す
        train_loss = train_one_epoch(model, train_dataset, optimizer, loss_fn, device)  # 学习一轮 / 1epoch学習する
        print(f"epoch {epoch:02d}: train_loss={train_loss:.6f}")  # 输出训练损失 / 訓練損失を出力する

    dev_accuracy = evaluate_accuracy_single(model, dev_dataset, device)  # 计算开发集正解率 / 開発セット正解率を計算する
    print(f"dev accuracy: {dev_accuracy:.6f}")  # 输出正解率 / 正解率を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

r'''
运行结果: / 実行結果:
==================================================
Knock 74: Accuracy
==================================================
device: cuda
epoch 01: train_loss=0.432879
epoch 02: train_loss=0.403548
epoch 03: train_loss=0.401023
epoch 04: train_loss=0.400191
epoch 05: train_loss=0.399824
dev accuracy: 0.780963
'''
