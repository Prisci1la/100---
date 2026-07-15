'''
knock99.py: 選好チューニング / 選好チューニング

正解ラベル応答をchosen、不正解ラベル応答をrejectedとして、TRLのDPOTrainerで選好チューニングする。
/ 正解ラベル応答をchosen、不正解ラベル応答をrejectedとして、TRLのDPOTrainerで選好チューニングする。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

from trl import DPOConfig, DPOTrainer  # DPO相关类 / DPO関連クラス

from chapter12_utils import DEFAULT_DPO_DIR, DEFAULT_TRAIN_PATH, MODEL_NAME, build_4bit_config, build_lora_config, load_tokenizer, make_preference_rows, read_sst2_rows, to_hf_dataset  # 导入共用工具 / 共通ツールを導入する


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock99: DPO preference tuning with TRL")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名 / モデル名
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="SST-2 train.tsv path")  # 训练数据 / 訓練データ
    parser.add_argument("--output-dir", default=str(DEFAULT_DPO_DIR), help="output directory")  # 输出目录 / 出力ディレクトリ
    parser.add_argument("--epochs", type=float, default=1.0, help="number of epochs")  # epoch数 / epoch数
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")  # batch大小 / batchサイズ
    parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")  # 学习率 / 学習率
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
    rows = read_sst2_rows(args.train_path, args.max_train_examples)  # 读取训练数据 / 訓練データを読む
    dataset = to_hf_dataset(make_preference_rows(rows))  # 创建DPO数据 / DPOデータを作る
    dpo_args = DPOConfig(  # 创建DPO训练参数 / DPO学習引数を作る
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        max_length=256,
        model_init_kwargs={"quantization_config": build_4bit_config(args.compute_dtype), "device_map": "auto"} if args.load_in_4bit else None,
    )
    trainer = DPOTrainer(  # 创建DPOTrainer / DPOTrainerを作る
        model=args.model_name,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout) if args.use_lora else None,
    )
    trainer.train()  # 执行DPO训练 / DPO学習を実行する
    trainer.save_model(args.output_dir)  # 保存模型 / モデルを保存する
    tokenizer.save_pretrained(args.output_dir)  # 保存tokenizer / tokenizerを保存する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ

