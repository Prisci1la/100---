'''
knock99.py: 選好チューニング / 選好チューニング

正解ラベル応答をchosen、不正解ラベル応答をrejectedとして、TRLのDPOTrainerで選好チューニングする。
/ 正解ラベル応答をchosen、不正解ラベル応答をrejectedとして、TRLのDPOTrainerで選好チューニングする。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # PyTorch / PyTorch
import torch.nn.functional as F  # 损失函数 / 損失関数
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式训练 / 分散学習
from torch.utils.data import DataLoader  # 数据加载 / データ読み込み
from torch.utils.data.distributed import DistributedSampler  # 分布式采样 / 分散サンプラー
from tqdm.auto import tqdm  # 进度条 / 進捗バー

from chapter12_utils import DEFAULT_DPO_DIR, DEFAULT_TRAIN_PATH, MODEL_NAME, PreferenceDataset, build_4bit_config, build_lora_config, collate_preferences, count_trainable_parameters, get_device_map, get_local_rank, load_causal_lm, load_tokenizer, make_preference_rows, read_sst2_rows, sequence_log_probability, set_trainable_parameters, should_use_legacy_torch, to_hf_dataset  # 导入共用工具 / 共通ツールを導入する


def add_bool_pair(parser, name, default, help_text):  # Python 3.7兼容的布尔参数 / Python 3.7互換の真偽引数
    dest = name.replace("-", "_")  # argparse保存名 / argparseの保存名
    group = parser.add_mutually_exclusive_group()  # 互斥组 / 排他グループ
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)  # 开启 / 有効化
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"disable {help_text}")  # 关闭 / 無効化
    parser.set_defaults(**{dest: default})  # 默认值 / 既定値


def setup_distributed():  # 初始化分布式 / 分散を初期化する
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))  # 进程数 / プロセス数
    if world_size > 1 and not torch.distributed.is_initialized():  # 需要DDP / DDPが必要
        torch.distributed.init_process_group(backend="nccl")  # 初始化NCCL / NCCLを初期化する
    local_rank = get_local_rank()  # 本地rank / ローカルrank
    if torch.cuda.is_available():  # 设置GPU / GPUを設定する
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return world_size, local_rank, device  # 返回分布式信息 / 分散情報を返す


def move_pair_batch(batch, device):  # 嵌套batch移动设备 / 入れ子batchをデバイスへ移す
    return {
        side: {key: value.to(device) for key, value in values.items()}
        for side, values in batch.items()
    }


def unwrap_model(model):  # 取出DDP内部模型 / DDP内部モデルを取り出す
    return model.module if hasattr(model, "module") else model


def legacy_train(args, tokenizer, rows):  # torch 1.5兼容DPO / torch 1.5互換DPO
    world_size, local_rank, device = setup_distributed()  # 初始化设备 / デバイスを初期化する
    policy = load_causal_lm(args.model_name, device=device, load_in_4bit=False)  # 策略模型 / 方策モデル
    reference = load_causal_lm(args.model_name, device=device, load_in_4bit=False).eval()  # 参考模型 / 参照モデル
    for parameter in reference.parameters():  # 冻结参考模型 / 参照モデルを凍結する
        parameter.requires_grad = False
    set_trainable_parameters(policy, args.trainable)  # 设置训练范围 / 学習範囲を設定する
    trainable, total = count_trainable_parameters(policy)  # 统计参数 / パラメータ統計
    dataset = PreferenceDataset(make_preference_rows(rows), tokenizer)  # 创建偏好Dataset / 選好Datasetを作る
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None  # 分布式采样 / 分散サンプラー
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=lambda examples: collate_preferences(examples, tokenizer),
    )
    policy.train()  # 训练模式 / 学習モード
    if world_size > 1:  # DDP包装 / DDPで包む
        policy = DDP(policy, device_ids=[local_rank] if device.type == "cuda" else None, broadcast_buffers=False)
    optimizer = torch.optim.AdamW([parameter for parameter in policy.parameters() if parameter.requires_grad], lr=args.lr)  # 优化器 / 最適化器
    if local_rank == 0:
        print(f"legacy_torch: torch={torch.__version__}, trainable={trainable}/{total}, world_size={world_size}, beta={args.beta}")
    global_step = 0  # 全局步数 / グローバルstep
    optimizer.zero_grad()  # 清梯度 / 勾配を消す
    for epoch in range(1, int(args.epochs) + 1):  # epoch循环 / epochループ
        if sampler is not None:
            sampler.set_epoch(epoch)  # DDP shuffle / DDP shuffle
        progress = tqdm(loader, desc=f"epoch {epoch}", disable=local_rank != 0)  # 进度条 / 進捗バー
        for step, batch in enumerate(progress, 1):  # batch循环 / batchループ
            batch = move_pair_batch(batch, device)  # 移动设备 / デバイスへ移す
            chosen_logps = sequence_log_probability(policy, batch["chosen"])  # chosen策略logp / chosen方策logp
            rejected_logps = sequence_log_probability(policy, batch["rejected"])  # rejected策略logp / rejected方策logp
            with torch.no_grad():  # 参考模型不求梯度 / 参照モデルは勾配なし
                ref_chosen_logps = sequence_log_probability(reference, batch["chosen"])  # chosen参考logp / chosen参照logp
                ref_rejected_logps = sequence_log_probability(reference, batch["rejected"])  # rejected参考logp / rejected参照logp
            pi_logratios = chosen_logps - rejected_logps  # 策略偏好差 / 方策の選好差
            ref_logratios = ref_chosen_logps - ref_rejected_logps  # 参考偏好差 / 参照の選好差
            loss = -F.logsigmoid(args.beta * (pi_logratios - ref_logratios)).mean() / args.gradient_accumulation_steps  # DPO loss / DPO loss
            loss.backward()  # 反传 / 逆伝播
            if step % args.gradient_accumulation_steps == 0 or step == len(loader):  # 梯度累积结束 / 勾配累積終了
                optimizer.step()  # 更新 / 更新
                optimizer.zero_grad()  # 清梯度 / 勾配を消す
                global_step += 1
            if local_rank == 0:
                progress.set_postfix(loss=float(loss.item() * args.gradient_accumulation_steps), step=global_step)
    if local_rank == 0:
        unwrap_model(policy).save_pretrained(args.output_dir)  # 保存模型 / モデルを保存する
        tokenizer.save_pretrained(args.output_dir)  # 保存tokenizer / tokenizerを保存する


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
    add_bool_pair(parser, "load-in-4bit", True, "load GPT2 in 4bit")  # 4bit载入 / 4bit読み込み
    add_bool_pair(parser, "use-lora", True, "train LoRA adapters")  # LoRA开关 / LoRA切替
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")  # LoRA rank / LoRA rank
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")  # LoRA alpha / LoRA alpha
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")  # LoRA dropout / LoRA dropout
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default="float16", help="4bit compute dtype")  # 计算dtype / 計算dtype
    parser.add_argument("--legacy-torch", choices=["auto", "on", "off"], default="auto", help="use torch 1.5 compatible manual DPO loop")  # 旧torch兼容 / 旧torch互換
    parser.add_argument("--trainable", choices=["last-block", "head", "all"], default="last-block", help="trainable parameters for legacy mode")  # 训练范围 / 学習範囲
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta for legacy mode")  # DPO beta / DPO beta
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    if args.load_in_4bit and not args.use_lora:  # 4bit训练必须配合LoRA / 4bit学習はLoRAと組み合わせる
        parser.error("--load-in-4bit requires --use-lora for trainable adapters")  # 明确报错 / 明確にエラーを出す
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読む
    rows = read_sst2_rows(args.train_path, args.max_train_examples)  # 读取训练数据 / 訓練データを読む
    force_legacy = {"auto": None, "on": True, "off": False}[args.legacy_torch]  # 旧torch模式 / 旧torchモード
    if should_use_legacy_torch(force_legacy):  # torch 1.5兼容路径 / torch 1.5互換経路
        print(f"device: manual, train: {len(rows)}, epochs: {args.epochs}, batch_size: {args.batch_size}, grad_accum: {args.gradient_accumulation_steps}")
        legacy_train(args, tokenizer, rows)  # 手写DPO / 手書きDPO
        return
    from trl import DPOConfig, DPOTrainer  # DPO相关类 / DPO関連クラス
    print(f"device: auto, train: {len(rows)}, epochs: {args.epochs}, batch_size: {args.batch_size}, grad_accum: {args.gradient_accumulation_steps}")  # 输出设置 / 設定を出力する
    dataset = to_hf_dataset(make_preference_rows(rows))  # 创建DPO数据 / DPOデータを作る
    dpo_args = DPOConfig(  # 创建DPO训练参数 / DPO学習引数を作る
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        disable_tqdm=False,
        report_to="none",
        max_length=256,
        ddp_find_unused_parameters=False,
        model_init_kwargs={"quantization_config": build_4bit_config(args.compute_dtype), "device_map": get_device_map()} if args.load_in_4bit else None,
    )
    try:
        trainer = DPOTrainer(  # 创建DPOTrainer / DPOTrainerを作る
            model=args.model_name,
            args=dpo_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout) if args.use_lora else None,
        )
    except TypeError:
        trainer = DPOTrainer(  # 旧TRL兼容 / 旧TRL互換
            model=args.model_name,
            args=dpo_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout) if args.use_lora else None,
        )
    trainer.train()  # 执行DPO训练 / DPO学習を実行する
    trainer.save_model(args.output_dir)  # 保存模型 / モデルを保存する
    tokenizer.save_pretrained(args.output_dir)  # 保存tokenizer / tokenizerを保存する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ


# DPO used the single-process fallback path after the multi-GPU legacy issue seen in knock98.
# progress excerpt: epoch 1 reached 1000/1000 in about 02:08, final displayed speed about 7.80 it/s, loss=0.807, step=125.
# near the end, loss values mostly stayed around 0.6-0.8 with occasional spikes.
# model saved to Chapter 12/models/dpo_sentiment_gpt2.
