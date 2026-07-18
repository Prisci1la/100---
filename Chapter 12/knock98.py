'''
knock98.py: ファインチューニング / ファインチューニング

感情分析promptに対して正解ラベルを返すようにGPT2をSFTする。
/ 感情分析promptに対して正解ラベルを返すようにGPT2をSFTする。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments  # Trainer相关类 / Trainer関連クラス

from chapter12_utils import DEFAULT_SFT_DIR, DEFAULT_TRAIN_PATH, MODEL_NAME, apply_lora, load_causal_lm, load_tokenizer, make_lm_rows, read_sst2_rows, to_hf_dataset  # 导入共用工具 / 共通ツールを導入する


def tokenize_dataset(dataset, tokenizer):  # 对SFT数据tokenize / SFTデータをtokenizeする
    def encode(batch):  # 定义批量编码函数 / バッチ符号化関数を定義する
        return tokenizer(batch["text"], truncation=True, max_length=128)  # 编码文本 / テキストを符号化する
    return dataset.map(encode, batched=True, remove_columns=["text"])  # 返回tokenized dataset / tokenized datasetを返す


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock98: SFT GPT2 for sentiment labels")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名 / モデル名
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="SST-2 train.tsv path")  # 训练数据 / 訓練データ
    parser.add_argument("--output-dir", default=str(DEFAULT_SFT_DIR), help="output directory")  # 输出目录 / 出力ディレクトリ
    parser.add_argument("--epochs", type=float, default=1.0, help="number of epochs")  # epoch数 / epoch数
    parser.add_argument("--batch-size", type=int, default=2, help="batch size")  # batch大小 / batchサイズ
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")  # 学习率 / 学習率
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="gradient accumulation steps")  # 梯度累积 / 勾配累積
    parser.add_argument("--max-train-examples", type=int, default=None, help="limit train examples")  # 训练上限 / 訓練上限
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True, help="load GPT2 in 4bit")  # 4bit载入 / 4bit読み込み
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True, help="train LoRA adapters")  # LoRA开关 / LoRA切替
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")  # LoRA rank / LoRA rank
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")  # LoRA alpha / LoRA alpha
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")  # LoRA dropout / LoRA dropout
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default="float16", help="4bit compute dtype")  # 计算dtype / 計算dtype
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    if args.load_in_4bit and not args.use_lora:  # 4bit训练必须配合LoRA / 4bit学習はLoRAと組み合わせる
        parser.error("--load-in-4bit requires --use-lora for trainable adapters")  # 明确报错 / 明確にエラーを出す
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読む
    model = load_causal_lm(args.model_name, load_in_4bit=args.load_in_4bit, compute_dtype=args.compute_dtype)  # 读取模型 / モデルを読む
    if args.use_lora:  # 使用LoRA训练 / LoRAで学習する
        model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, prepare_for_kbit=args.load_in_4bit)  # 注入LoRA / LoRAを注入する
        model.print_trainable_parameters()  # 输出可训练参数 / 学習可能パラメータを出力する
    rows = read_sst2_rows(args.train_path, args.max_train_examples)  # 读取训练数据 / 訓練データを読む
    print(f"device: auto, train: {len(rows)}, epochs: {args.epochs}, batch_size: {args.batch_size}, grad_accum: {args.gradient_accumulation_steps}")  # 输出设置 / 設定を出力する
    dataset = tokenize_dataset(to_hf_dataset(make_lm_rows(rows)), tokenizer)  # 构造并tokenize数据 / データを作ってtokenizeする
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # 因果LM用collator / 因果LM用collator
    training_args = TrainingArguments(  # 创建训练参数 / 学習引数を作る
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_strategy="epoch",
        logging_steps=10,
        disable_tqdm=False,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator, processing_class=tokenizer)  # 创建Trainer / Trainerを作る
    trainer.train()  # 执行训练 / 学習を実行する
    trainer.save_model(args.output_dir)  # 保存模型 / モデルを保存する
    tokenizer.save_pretrained(args.output_dir)  # 保存tokenizer / tokenizerを保存する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ

