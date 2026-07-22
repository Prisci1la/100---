'''
knock98.py: ファインチューニング / ファインチューニング

感情分析promptに対して正解ラベルを返すようにGPT2をSFTする。
/ 感情分析promptに対して正解ラベルを返すようにGPT2をSFTする。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # PyTorch / PyTorch
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式训练 / 分散学習
from torch.utils.data import DataLoader, Dataset  # 数据加载 / データ読み込み
from torch.utils.data.distributed import DistributedSampler  # 分布式采样 / 分散サンプラー
from tqdm.auto import tqdm  # 进度条 / 進捗バー
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # Transformers自动类 / Transformers自動クラス

from chapter12_utils import DEFAULT_SFT_DIR, DEFAULT_TRAIN_PATH, MODEL_NAME, apply_lora, get_local_rank, read_sst2_rows, should_use_legacy_torch, to_hf_dataset  # 导入共用工具 / 共通ツールを導入する


def load_tokenizer(model_name=MODEL_NAME):  # 读取tokenizer / tokenizerを読み込む
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 从Hugging Face读取 / Hugging Faceから読む
    if tokenizer.pad_token is None:  # GPT2默认没有PAD / GPT2は既定でPADを持たない
        tokenizer.pad_token = tokenizer.eos_token  # 用EOS作为PAD / EOSをPADとして使う
    return tokenizer  # 返回tokenizer / tokenizerを返す


def get_torch_dtype(dtype_name="float16"):  # 字符串转torch dtype / 文字列をtorch dtypeへ変換する
    mapping = {"float16": torch.float16, "bfloat16": getattr(torch, "bfloat16", torch.float16), "float32": torch.float32}
    return mapping[dtype_name]  # 返回dtype / dtypeを返す


def build_4bit_config(compute_dtype="float16"):  # 构建4bit量化配置 / 4bit量子化設定を作る
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=get_torch_dtype(compute_dtype),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def get_device_map():  # 量化模型设备配置 / 量子化モデルのデバイス配置
    if torch.cuda.is_available() and "LOCAL_RANK" in __import__("os").environ:
        return {"": get_local_rank()}
    return "auto"


def load_causal_lm(model_name=MODEL_NAME, device=None, load_in_4bit=False, compute_dtype="float16"):  # 读取因果语言模型 / 因果言語モデルを読み込む
    if load_in_4bit:  # 4bit量化加载 / 4bit量子化で読む
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=build_4bit_config(compute_dtype),
            device_map=get_device_map(),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)  # 普通加载 / 通常読み込み
    model.config.pad_token_id = model.config.eos_token_id  # 设置PAD ID / PAD IDを設定する
    if load_in_4bit:
        return model  # device_map已经配置 / device_mapで配置済み
    return model.to(device)  # 移动到设备 / デバイスへ移す


def sentiment_prompt(text):  # 构造情感分析prompt / 感情分析promptを作る
    return f"Review: {text}\nSentiment:"  # 返回prompt / promptを返す


def label_text(label):  # 标签转文本 / ラベルをテキストへ変換する
    return " positive" if int(label) == 1 else " negative"  # 1为positive / 1はpositive


def make_lm_rows(rows):  # 构造SFT文本 / SFTテキストを作る
    return [{"text": sentiment_prompt(row["text"]) + label_text(row["label"])} for row in rows]  # prompt加正确标签 / promptに正解ラベルを足す


def masked_labels(input_ids, attention_mask):  # 为LM构造忽略padding的labels / LM用labelsを作る
    labels = input_ids.clone()  # 复制输入ID / 入力IDをコピー
    labels[attention_mask == 0] = -100  # padding不计loss / paddingをlossから除外
    return labels  # 返回labels / labelsを返す


class LMTextDataset(Dataset):  # 手写SFT用Dataset / 手書きSFT用Dataset
    def __init__(self, rows, tokenizer, max_length=128):  # 初始化 / 初期化
        self.rows = rows  # 保存文本行 / テキスト行を保存
        self.tokenizer = tokenizer  # tokenizer / tokenizer
        self.max_length = max_length  # 最大长度 / 最大長

    def __len__(self):  # 样本数 / サンプル数
        return len(self.rows)

    def __getitem__(self, index):  # 取样本 / サンプルを取る
        return self.tokenizer(self.rows[index]["text"], truncation=True, max_length=self.max_length)  # 编码 / 符号化


def collate_lm_texts(examples, tokenizer):  # 手写SFT batch整理 / 手書きSFT batch整形
    batch = tokenizer.pad(examples, return_tensors="pt")  # padding / paddingする
    batch["labels"] = masked_labels(batch["input_ids"], batch["attention_mask"])  # 添加labels / labelsを足す
    return batch  # 返回batch / batchを返す


def set_trainable_parameters(model, mode="all"):  # 设置可训练范围 / 学習可能範囲を設定する
    if mode == "all":  # 全参数训练 / 全パラメータ学習
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    for parameter in model.parameters():  # 先冻结 / まず凍結する
        parameter.requires_grad = False
    if mode == "head":  # 只训练LM head / LM headだけ学習
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True
    elif mode == "last-block":  # 训练最后block和LM head / 最終blockとLM headを学習
        for parameter in model.transformer.h[-1].parameters():
            parameter.requires_grad = True
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"unknown trainable mode: {mode}")  # 未知模式 / 未知モード


def count_trainable_parameters(model):  # 统计可训练参数 / 学習可能パラメータを数える
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)  # 可训练数 / 学習可能数
    total = sum(parameter.numel() for parameter in model.parameters())  # 总数 / 総数
    return trainable, total  # 返回统计 / 統計を返す


def tokenize_dataset(dataset, tokenizer):  # 对SFT数据tokenize / SFTデータをtokenizeする
    def encode(batch):  # 定义批量编码函数 / バッチ符号化関数を定義する
        return tokenizer(batch["text"], truncation=True, max_length=128)  # 编码文本 / テキストを符号化する
    return dataset.map(encode, batched=True, remove_columns=["text"])  # 返回tokenized dataset / tokenized datasetを返す


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


def unwrap_model(model):  # 取出DDP内部模型 / DDP内部モデルを取り出す
    return model.module if hasattr(model, "module") else model


def save_model(model, tokenizer, output_dir, rank):  # 只在rank0保存 / rank0だけ保存する
    if rank != 0:
        return
    unwrap_model(model).save_pretrained(output_dir)  # 保存模型 / モデルを保存する
    tokenizer.save_pretrained(output_dir)  # 保存tokenizer / tokenizerを保存する


def legacy_train(args, tokenizer, rows):  # torch 1.5兼容SFT / torch 1.5互換SFT
    world_size, local_rank, device = setup_distributed()  # 初始化设备 / デバイスを初期化する
    model = load_causal_lm(args.model_name, device=device, load_in_4bit=False)  # 普通加载 / 通常読み込み
    set_trainable_parameters(model, args.trainable)  # 设置训练范围 / 学習範囲を設定する
    trainable, total = count_trainable_parameters(model)  # 统计参数 / パラメータ統計
    dataset = LMTextDataset(make_lm_rows(rows), tokenizer)  # 创建Dataset / Datasetを作る
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None  # 分布式采样 / 分散サンプラー
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=lambda examples: collate_lm_texts(examples, tokenizer),
    )
    model.train()  # 训练模式 / 学習モード
    if world_size > 1:  # DDP包装 / DDPで包む
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, broadcast_buffers=False)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr)  # 优化器 / 最適化器
    total_steps = max(1, len(loader) * int(args.epochs))  # 总步数 / 総step数
    if local_rank == 0:
        print(f"legacy_torch: torch={torch.__version__}, trainable={trainable}/{total}, world_size={world_size}, total_steps={total_steps}")
    global_step = 0  # 全局步数 / グローバルstep
    optimizer.zero_grad()  # 清梯度 / 勾配を消す
    for epoch in range(1, int(args.epochs) + 1):  # epoch循环 / epochループ
        if sampler is not None:
            sampler.set_epoch(epoch)  # DDP shuffle / DDP shuffle
        progress = tqdm(loader, desc=f"epoch {epoch}", disable=local_rank != 0)  # 进度条 / 進捗バー
        for step, batch in enumerate(progress, 1):  # batch循环 / batchループ
            batch = {key: value.to(device) for key, value in batch.items()}  # 移动设备 / デバイスへ移す
            loss = model(**batch).loss / args.gradient_accumulation_steps  # 前向loss / 順伝播loss
            loss.backward()  # 反传 / 逆伝播
            if step % args.gradient_accumulation_steps == 0 or step == len(loader):  # 梯度累积结束 / 勾配累積終了
                optimizer.step()  # 更新 / 更新
                optimizer.zero_grad()  # 清梯度 / 勾配を消す
                global_step += 1
            if local_rank == 0:
                progress.set_postfix(loss=float(loss.item() * args.gradient_accumulation_steps), step=global_step)
    save_model(model, tokenizer, args.output_dir, local_rank)  # 保存 / 保存


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
    add_bool_pair(parser, "load-in-4bit", True, "load GPT2 in 4bit")  # 4bit载入 / 4bit読み込み
    add_bool_pair(parser, "use-lora", True, "train LoRA adapters")  # LoRA开关 / LoRA切替
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")  # LoRA rank / LoRA rank
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")  # LoRA alpha / LoRA alpha
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")  # LoRA dropout / LoRA dropout
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default="float16", help="4bit compute dtype")  # 计算dtype / 計算dtype
    parser.add_argument("--legacy-torch", choices=["auto", "on", "off"], default="auto", help="use torch 1.5 compatible manual loop")  # 旧torch兼容 / 旧torch互換
    parser.add_argument("--trainable", choices=["last-block", "head", "all"], default="last-block", help="trainable parameters for legacy mode")  # 训练范围 / 学習範囲
    args = parser.parse_args()  # 解析参数 / 引数を解析する
    if args.load_in_4bit and not args.use_lora:  # 4bit训练必须配合LoRA / 4bit学習はLoRAと組み合わせる
        parser.error("--load-in-4bit requires --use-lora for trainable adapters")  # 明确报错 / 明確にエラーを出す
    tokenizer = load_tokenizer(args.model_name)  # 读取tokenizer / tokenizerを読む
    rows = read_sst2_rows(args.train_path, args.max_train_examples)  # 读取训练数据 / 訓練データを読む
    force_legacy = {"auto": None, "on": True, "off": False}[args.legacy_torch]  # 旧torch模式 / 旧torchモード
    if should_use_legacy_torch(force_legacy):  # torch 1.5兼容路径 / torch 1.5互換経路
        print(f"device: manual, train: {len(rows)}, epochs: {args.epochs}, batch_size: {args.batch_size}, grad_accum: {args.gradient_accumulation_steps}")
        legacy_train(args, tokenizer, rows)  # 手写训练 / 手書き学習
        return
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments  # Trainer相关类 / Trainer関連クラス
    model = load_causal_lm(args.model_name, load_in_4bit=args.load_in_4bit, compute_dtype=args.compute_dtype)  # 读取模型 / モデルを読む
    if args.use_lora:  # 使用LoRA训练 / LoRAで学習する
        model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, prepare_for_kbit=args.load_in_4bit)  # 注入LoRA / LoRAを注入する
        model.print_trainable_parameters()  # 输出可训练参数 / 学習可能パラメータを出力する
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
        ddp_find_unused_parameters=False,
    )
    try:
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator, processing_class=tokenizer)  # 创建Trainer / Trainerを作る
    except TypeError:
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator, tokenizer=tokenizer)  # 旧Transformers兼容 / 旧Transformers互換
    trainer.train()  # 执行训练 / 学習を実行する
    trainer.save_model(args.output_dir)  # 保存模型 / モデルを保存する
    tokenizer.save_pretrained(args.output_dir)  # 保存tokenizer / tokenizerを保存する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行時のみ動かす
    main()  # 调用主函数 / メイン関数を呼ぶ



# device: auto, train: 1000, epochs: 1.0, batch_size: 2, grad_accum: 8
# trainable params: LoRA adapters only; GPT2-medium was loaded with 4bit quantization.
# SFT text format used problem 96 prompt plus the correct label response:
# 01. Review: hide new secretions from the parental units
#     Sentiment: negative
# 02. Review: contains no wit , only labored gags
#     Sentiment: negative
# 03. Review: that loves its characters and communicates something rather beautiful about human nature
#     Sentiment: positive
# 04. Review: remains utterly satisfied to remain the same throughout
#     Sentiment: negative
# 05. Review: demonstrates that the director of such hollywood blockbusters as patriot games can still turn out a small , personal film with an emotional wallop .
#     Sentiment: positive
# first attempt: legacy 8-GPU DDP failed before completion.
# error: RuntimeError: Unsupported data type for NCCL process group.
# fallback: single-process SFT run completed successfully.
# train progress excerpt:
#   10/1000  loss about 3.60
#  100/1000  loss about 2.70
#  500/1000  loss about 2.45
# 1000/1000  final displayed loss around 2.37
# epoch 1 reached 1000/1000 in about 01:03, speed about 15.76 it/s, final step=125 after gradient accumulation.
# model saved to Chapter 12/models/sft_sentiment_gpt2.
