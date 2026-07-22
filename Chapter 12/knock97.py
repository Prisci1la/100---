'''
knock97.py: 埋め込みに基づく感情分析 / 埋め込みに基づく感情分析

GPT2の文ベクトルを固定特徴量として、線形分類器でSST-2極性を学習する。
/ GPT2の文ベクトルを固定特徴量として、線形分類器でSST-2極性を学習する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # 导入PyTorch / PyTorchを導入する
from torch import nn  # 神经网络模块 / ニューラルネットワーク
from torch.utils.data import DataLoader, Dataset  # DataLoaderを導入する / DataLoaderを導入する
from tqdm.auto import tqdm  # 进度条 / 進捗バー
from transformers import AutoModel, AutoTokenizer  # Transformers自动类 / Transformers自動クラス

from chapter12_utils import (  # 导入共用工具 / 共通ツールを導入する
    DEFAULT_DEV_PATH,
    DEFAULT_TRAIN_PATH,
    MODEL_NAME,
    get_device,
    read_sst2_rows,
)


def load_tokenizer(model_name=MODEL_NAME):  # 读取tokenizer / tokenizerを読み込む
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 从Hugging Face读取 / Hugging Faceから読む
    if tokenizer.pad_token is None:  # GPT2默认没有PAD / GPT2は既定でPADを持たない
        tokenizer.pad_token = tokenizer.eos_token  # 用EOS作为PAD / EOSをPADとして使う
    return tokenizer  # 返回tokenizer / tokenizerを返す


def load_encoder(model_name=MODEL_NAME, device=None):  # 读取GPT2主体 / GPT2本体を読む
    model = AutoModel.from_pretrained(model_name)  # 读取无LM头模型 / LMヘッドなしモデルを読む
    return model.to(device)  # 移动到设备 / デバイスへ移す


class GPT2EmbeddingDataset(Dataset):  # GPT2埋め込み分類用Dataset / GPT2埋め込み分類用Dataset
    def __init__(self, rows, tokenizer, max_length=128):  # 初始化Dataset / Datasetを初期化する
        self.rows = rows  # 保存样本 / サンプルを保存する
        self.tokenizer = tokenizer  # 保存tokenizer / tokenizerを保存する
        self.max_length = max_length  # 最大长度 / 最大長

    def __len__(self):  # 样本数 / サンプル数
        return len(self.rows)

    def __getitem__(self, index):  # 取一个样本 / 1サンプルを取る
        row = self.rows[index]  # 取行 / 行を取る
        enc = self.tokenizer(row["text"], truncation=True, max_length=self.max_length)  # 编码文本 / テキストを符号化する
        enc["labels"] = row["label"]  # 添加标签 / ラベルを追加する
        return enc  # 返回样本 / サンプルを返す


def collate_gpt2_classifier(examples, tokenizer):  # 分类任务batch整理 / 分類タスクbatchを整える
    features = [{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in examples]  # 取输入特征 / 入力特徴を取る
    batch = tokenizer.pad(features, return_tensors="pt")  # padding / paddingする
    batch["labels"] = torch.tensor([e["labels"] for e in examples], dtype=torch.long)  # 标签tensor / ラベルtensor
    return batch  # 返回batch / batchを返す


class GPT2EmbeddingClassifier(nn.Module):  # GPT2埋め込み+線形分類器 / GPT2埋め込み+線形分類器
    def __init__(self, encoder, hidden_size):  # 初始化模型 / モデルを初期化する
        super().__init__()  # 父类初始化 / 親クラス初期化
        self.encoder = encoder  # GPT2主体 / GPT2本体
        self.classifier = nn.Linear(hidden_size, 2)  # 线性分类层 / 線形分類層
        for parameter in self.encoder.parameters():  # 固定GPT2 / GPT2を固定する
            parameter.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):  # 前向计算 / 順伝播
        with torch.no_grad():  # 固定encoder不求梯度 / encoderは勾配なし
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)  # 运行GPT2 / GPT2を実行する
        hidden = outputs.last_hidden_state  # token向量 / tokenベクトル
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1  # 最后有效token位置 / 最後の有効token位置
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]  # 取最后有效token向量 / 最後の有効tokenベクトル
        logits = self.classifier(pooled)  # 分类logits / 分類logits
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None  # loss / loss
        return loss, logits  # 返回loss和logits / lossとlogitsを返す


def evaluate(model, loader, device):  # 评价分类器 / 分類器を評価する
    model.eval()  # 切换评价模式 / 評価モードに切り替える
    correct = 0  # 正确数 / 正解数
    total = 0  # 总数 / 総数
    with torch.no_grad():  # 不计算梯度 / 勾配を計算しない
        for batch in tqdm(loader, desc="eval", leave=False):  # 遍历batch / batchを走査する
            labels = batch.pop("labels").to(device)  # 取出标签 / ラベルを取り出す
            batch = {key: value.to(device) for key, value in batch.items()}  # 移动输入 / 入力を移動する
            _loss, logits = model(**batch, labels=labels)  # 前向计算 / 順伝播を行う
            correct += (logits.argmax(dim=-1) == labels).sum().item()  # 累加正确数 / 正解数を加算する
            total += labels.numel()  # 累加总数 / 総数を加算する
    return correct / max(total, 1)  # 返回正解率 / 正解率を返す


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock97: GPT2 embedding classifier")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名 / モデル名
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="SST-2 train.tsv path")  # 训练数据 / 訓練データ
    parser.add_argument("--dev-path", default=str(DEFAULT_DEV_PATH), help="SST-2 dev.tsv path")  # 开发数据 / 開発データ
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs")  # epoch数 / epoch数
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")  # batch大小 / batchサイズ
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")  # 学习率 / 学習率
    parser.add_argument("--max-train-examples", type=int, default=None, help="limit train examples")  # 训练上限 / 訓練上限
    parser.add_argument("--max-dev-examples", type=int, default=None, help="limit dev examples")  # 开发上限 / 開発上限
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    device = get_device()  # 获取设备 / デバイスを取得する
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読む
    encoder = load_encoder(args.model_name, device=device).eval()  # 读取GPT2主体 / GPT2本体を読む
    model = GPT2EmbeddingClassifier(encoder, encoder.config.hidden_size).to(device)  # 创建分类模型 / 分類モデルを作る
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)  # 只优化分类层 / 分類層だけを最適化する
    train_rows = read_sst2_rows(args.train_path, args.max_train_examples)  # 读取训练数据 / 訓練データを読む
    dev_rows = read_sst2_rows(args.dev_path, args.max_dev_examples)  # 读取开发数据 / 開発データを読む
    train_loader = DataLoader(GPT2EmbeddingDataset(train_rows, tokenizer), batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_gpt2_classifier(x, tokenizer))  # 训练loader / 訓練loader
    dev_loader = DataLoader(GPT2EmbeddingDataset(dev_rows, tokenizer), batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_gpt2_classifier(x, tokenizer))  # 开发loader / 開発loader
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 97: Embedding-based Sentiment")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(f"device: {device}, train: {len(train_rows)}, dev: {len(dev_rows)}, batch_size: {args.batch_size}")  # 输出设置 / 設定を出力する
    for epoch in range(1, args.epochs + 1):  # 按epoch循环 / epochごとに繰り返す
        model.train()  # 训练模式 / 学習モード
        total_loss = 0.0  # 累计loss / lossを累積する
        for batch in tqdm(train_loader, desc="train", leave=False):  # 遍历batch / batchを走査する
            labels = batch.pop("labels").to(device)  # 取标签 / ラベルを取る
            batch = {key: value.to(device) for key, value in batch.items()}  # 移动输入 / 入力を移動する
            optimizer.zero_grad()  # 清空梯度 / 勾配を消す
            loss, _logits = model(**batch, labels=labels)  # 前向计算 / 順伝播を行う
            loss.backward()  # 反向传播 / 逆伝播を行う
            optimizer.step()  # 更新参数 / パラメータを更新する
            total_loss += loss.item()  # 累加loss / lossを加算する
        accuracy = evaluate(model, dev_loader, device)  # 评价 / 評価する
        print(f"epoch {epoch:02d}: train_loss={total_loss / max(len(train_loader), 1):.6f}, dev_accuracy={accuracy:.6f}")  # 输出进度 / 進捗を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ



# device: cuda; train=67349, dev=872, batch_size=16.
# progress: each epoch ran 4210 train batches, around 20-25 it/s after warm-up on TITAN RTX.
# epoch 01: train_loss=0.420917, dev_accuracy=0.865826
# epoch 02: train_loss=0.398262, dev_accuracy=0.869266
# epoch 03: train_loss=0.393835, dev_accuracy=0.880734
# Accuracy improved across all 3 epochs, final dev_accuracy=88.07%.
