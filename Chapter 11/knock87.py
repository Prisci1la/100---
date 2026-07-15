'''
knock87.py: ファインチューニング / ファインチューニング

BERTを極性分析向けにファインチューニングし、全epoch中のbest valid scoreのモデルを保存する。
/ BERTを極性分析向けにファインチューニングし、全epoch中のbest valid scoreのモデルを保存する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # 导入PyTorch / PyTorchを導入する
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification

from chapter11_utils import (  # 导入共用工具 / 共通ツールを導入する
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DEV_PATH,
    DEFAULT_TRAIN_PATH,
    MODEL_NAME,
    configure_quiet_mode,
    create_data_loader,
    create_dataset,
    get_device,
    load_tokenizer,
    save_standard_model,
)


def move_batch_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_sequence_classifier(model_name, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return model.to(device)


def train_one_epoch(model, data_loader, optimizer, device, show_progress=False):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in tqdm(data_loader, desc="train", leave=False, disable=not show_progress):
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("labels")
        batch.pop("texts", None)
        optimizer.zero_grad()
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


def evaluate(model, data_loader, device, show_progress=False):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="eval", leave=False, disable=not show_progress):
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            batch.pop("texts", None)
            outputs = model(**batch, labels=labels)
            predictions = outputs.logits.argmax(dim=-1)
            batch_size = labels.size(0)
            correct += (predictions == labels).sum().item()
            total_loss += outputs.loss.item() * batch_size
            total_examples += batch_size
    return {"loss": total_loss / max(total_examples, 1), "accuracy": correct / max(total_examples, 1)}


def fine_tune(model, tokenizer, train_loader, dev_loader, optimizer, device, epochs, output_dir, metric_name):
    best_metric = None
    best_metrics = {}
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        dev_metrics = evaluate(model, dev_loader, device)
        current = dev_metrics[metric_name]
        if metric_name == "accuracy":
            better = best_metric is None or current > best_metric
        else:
            better = best_metric is None or current < best_metric
        if better:
            best_metric = current
            best_metrics = {"epoch": epoch, "train_loss": train_loss, **dev_metrics, "selection_metric": metric_name}
            save_standard_model(model, tokenizer, output_dir, best_metrics)
        print(f"epoch {epoch:02d}: dev_accuracy={dev_metrics['accuracy']:.6f}")
    return best_metrics


def main():  # 定义主函数 / メイン関数を定義する
    configure_quiet_mode()  # 抑制额外日志 / 余計なログを抑制する
    parser = argparse.ArgumentParser(description="knock87: fine-tune BERT on SST-2")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名称 / モデル名
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="SST-2 train.tsv path")  # 训练数据路径 / 訓練データパス
    parser.add_argument("--dev-path", default=str(DEFAULT_DEV_PATH), help="SST-2 dev.tsv path")  # 开发数据路径 / 開発データパス
    parser.add_argument("--output-dir", default=str(DEFAULT_CHECKPOINT_DIR), help="best model output directory")  # 保存目录 / 保存ディレクトリ
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs")  # epoch数 / epoch数
    parser.add_argument("--batch-size", type=int, default=16, help="mini-batch size")  # batch大小 / batchサイズ
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")  # 学习率 / 学習率
    parser.add_argument("--max-length", type=int, default=128, help="maximum token length")  # 最大长度 / 最大長
    parser.add_argument("--metric-name", choices=["accuracy", "loss"], default="accuracy", help="best model selection metric")  # 最佳模型指标 / 最良モデル選択指標
    parser.add_argument("--max-train-examples", type=int, default=None, help="limit train examples for smoke test")  # 训练样本上限 / 訓練サンプル上限
    parser.add_argument("--max-dev-examples", type=int, default=None, help="limit dev examples for smoke test")  # 开发样本上限 / 開発サンプル上限
    args = parser.parse_args()  # 解析命令行参数 / コマンドライン引数を解析する

    device = get_device()  # 获取设备 / デバイスを取得する
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読み込む
    train_dataset = create_dataset(args.train_path, tokenizer, args.max_length, args.max_train_examples)  # 创建训练Dataset / 訓練Datasetを作る
    dev_dataset = create_dataset(args.dev_path, tokenizer, args.max_length, args.max_dev_examples)  # 创建开发Dataset / 開発Datasetを作る
    train_loader = create_data_loader(train_dataset, tokenizer, args.batch_size, shuffle=True)  # 创建训练DataLoader / 訓練DataLoaderを作る
    dev_loader = create_data_loader(dev_dataset, tokenizer, args.batch_size, shuffle=False)  # 创建开发DataLoader / 開発DataLoaderを作る
    model = load_sequence_classifier(args.model_name, device=device)  # 读取BERT分类模型 / BERT分類モデルを読み込む
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 创建AdamW优化器 / AdamW最適化器を作る

    best_metrics = fine_tune(model, tokenizer, train_loader, dev_loader, optimizer, device, args.epochs, args.output_dir, args.metric_name)  # 执行微调 / 微調整を実行する
    print(f"best_epoch: {best_metrics['epoch']}")  # 输出最佳epoch / 最良epochを出力する
    print(f"best_dev_accuracy: {best_metrics['accuracy']:.6f}")  # 输出最佳验证精度 / 最良検証正解率を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
Execution result:
epoch 01: dev_accuracy=0.917431
epoch 02: dev_accuracy=0.923165
epoch 03: dev_accuracy=0.910550
best_epoch: 2
best_dev_accuracy: 0.923165

Generated readable file: checkpoints/best_model/metrics.json
{
  "epoch": 2,
  "train_loss": 0.10749771991787312,
  "loss": 0.212511660138128,
  "accuracy": 0.9231651376146789,
  "selection_metric": "accuracy"
}

Generated readable file head: checkpoints/best_model/config.json
{
  "add_cross_attention": false,
  "architectures": [
    "BertForSequenceClassification"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": null,
  "classifier_dropout": null,
  "dtype": "float32",
  "eos_token_id": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "is_decoder": false,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "problem_type": "single_label_classification",
  "tie_word_embeddings": true,
  "transformers_version": "5.12.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
'''
