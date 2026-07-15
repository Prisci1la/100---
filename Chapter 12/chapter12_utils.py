'''
chapter12_utils.py: 第12回共用工具 / 第12回の共通ユーティリティ

GPT2-mediumの生成、確率計算、SST-2評価、SFT、DPO用データ作成をまとめる。
/ GPT2-mediumの生成、確率計算、SST-2評価、SFT、DPO用データ作成をまとめる。
'''

from __future__ import annotations  # 延迟类型注解的求值 / 型注釈の評価を遅延する

import csv  # TSV读取库 / TSV読み込みライブラリ
import math  # 数学函数 / 数学関数
from pathlib import Path  # 路径处理类 / パス処理クラス

import torch  # 导入PyTorch / PyTorchを導入する
from datasets import Dataset as HFDataset  # Hugging Face Dataset / Hugging Face Dataset
from torch import nn  # 神经网络模块 / ニューラルネットワークモジュール
from torch.utils.data import DataLoader, Dataset  # PyTorch数据工具 / PyTorchデータツール
from tqdm.auto import tqdm  # 进度条 / 進捗バー
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # Transformers自动类 / Transformers自動クラス


BASE_DIR = Path(__file__).resolve().parent  # 当前第12回目录 / 現在の第12回ディレクトリ
MODEL_NAME = "openai-community/gpt2-medium"  # 指定GPT2-medium模型 / GPT2-mediumモデルを指定する
PROMPT = "The movie was full of"  # 问题90-92的提示 / 問題90-92のプロンプト
CHAT_QUESTION = "What do you call a sweet eaten after dinner?"  # 问题94的提问 / 問題94の質問
FOLLOWUP_QUESTION = "Please give me the plural form of the word with its spelling in reverse order."  # 问题95追加提问 / 問題95の追加質問
DEFAULT_DEV_PATH = BASE_DIR.parent / "Chapter 8" / "data" / "SST-2" / "dev.tsv"  # SST-2开发集 / SST-2開発セット
DEFAULT_TRAIN_PATH = BASE_DIR.parent / "Chapter 8" / "data" / "SST-2" / "train.tsv"  # SST-2训练集 / SST-2訓練セット
DEFAULT_SFT_DIR = BASE_DIR / "models" / "sft_sentiment_gpt2"  # SFT保存目录 / SFT保存ディレクトリ
DEFAULT_DPO_DIR = BASE_DIR / "models" / "dpo_sentiment_gpt2"  # DPO保存目录 / DPO保存ディレクトリ
PPL_SENTENCES = [  # 问题93的例句 / 問題93の例文
    "The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises",
]


def get_device():  # 获取可用设备 / 利用可能なデバイスを取得する
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有GPU就使用CUDA / GPUがあればCUDAを使う


def load_tokenizer(model_name=MODEL_NAME):  # 读取tokenizer / tokenizerを読み込む
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 从Hugging Face读取 / Hugging Faceから読む
    if tokenizer.pad_token is None:  # GPT2默认没有PAD / GPT2は既定でPADを持たない
        tokenizer.pad_token = tokenizer.eos_token  # 使用EOS作为PAD / EOSをPADとして使う
    return tokenizer  # 返回tokenizer / tokenizerを返す


def get_torch_dtype(dtype_name="float16"):  # 字符串转torch dtype / 文字列をtorch dtypeへ変換する
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}  # 可选dtype / 選択可能dtype
    return mapping[dtype_name]  # 返回dtype / dtypeを返す


def build_4bit_config(compute_dtype="float16"):  # 构建4bit量化配置 / 4bit量子化設定を作る
    return BitsAndBytesConfig(  # bitsandbytes 4bit配置 / bitsandbytes 4bit設定
        load_in_4bit=True,
        bnb_4bit_compute_dtype=get_torch_dtype(compute_dtype),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(r=16, alpha=32, dropout=0.05):  # 构建GPT2用LoRA配置 / GPT2用LoRA設定を作る
    from peft import LoraConfig, TaskType  # 延迟导入PEFT / PEFTを遅延importする

    return LoraConfig(  # GPT2的线性层是Conv1D，需要fan_in_fan_out / GPT2の線形層はConv1Dなのでfan_in_fan_out
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["c_attn", "c_proj", "c_fc"],
        fan_in_fan_out=True,
    )


def apply_lora(model, r=16, alpha=32, dropout=0.05, prepare_for_kbit=False):  # 给模型加LoRA / モデルにLoRAを追加する
    from peft import get_peft_model, prepare_model_for_kbit_training  # 延迟导入PEFT / PEFTを遅延importする

    if prepare_for_kbit:  # 4bit训练前的准备 / 4bit学習前の準備
        model = prepare_model_for_kbit_training(model)  # 开启输入梯度等设置 / 入力勾配などを設定する
    model.config.use_cache = False  # 训练时关闭cache / 学習時はcacheを止める
    model = get_peft_model(model, build_lora_config(r, alpha, dropout))  # 注入LoRA / LoRAを注入する
    return model  # 返回PEFT模型 / PEFTモデルを返す


def load_causal_lm(model_name=MODEL_NAME, device=None, load_in_4bit=False, compute_dtype="float16"):  # 读取因果语言模型 / 因果言語モデルを読み込む
    device = get_device() if device is None else device  # 自动选择设备 / デバイスを自動選択する
    if load_in_4bit:  # 4bit量化载入 / 4bit量子化で読み込む
        model = AutoModelForCausalLM.from_pretrained(  # 读取量化模型 / 量子化モデルを読み込む
            model_name,
            quantization_config=build_4bit_config(compute_dtype),
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)  # 读取模型 / モデルを読み込む
    model.config.pad_token_id = model.config.eos_token_id  # 设置PAD ID / PAD IDを設定する
    if load_in_4bit:  # 量化模型由device_map放置 / 量子化モデルはdevice_mapが配置する
        return model  # 返回模型 / モデルを返す
    return model.to(device)  # 移动到设备 / デバイスへ移す


def load_encoder(model_name=MODEL_NAME, device=None):  # 读取GPT2编码器主体 / GPT2エンコーダ相当の本体を読む
    device = get_device() if device is None else device  # 自动选择设备 / デバイスを自動選択する
    model = AutoModel.from_pretrained(model_name)  # 读取无LM头模型 / LMヘッドなしモデルを読む
    return model.to(device)  # 移动到设备 / デバイスへ移す


def next_token_topk(tokenizer, model, prompt=PROMPT, top_k=10, device=None):  # 预测下一个token top-k / 次token top-kを予測する
    device = get_device() if device is None else device  # 自动选择设备 / デバイスを自動選択する
    encoding = tokenizer(prompt, return_tensors="pt").to(device)  # 编码prompt / promptを符号化する
    with torch.no_grad():  # 不计算梯度 / 勾配を計算しない
        logits = model(**encoding).logits[0, -1]  # 取得最后位置logit / 最後位置のlogitを取得する
        probabilities = torch.softmax(logits, dim=-1)  # 转换为概率 / 確率へ変換する
        top_probs, top_ids = torch.topk(probabilities, k=top_k)  # 取top-k / top-kを取る
    tokens = [tokenizer.decode([token_id]) for token_id in encoding["input_ids"][0].tolist()]  # 确认prompt token列 / prompt token列を確認する
    predictions = [(tokenizer.decode([idx]).strip(), prob.item()) for idx, prob in zip(top_ids.tolist(), top_probs)]  # 转换为文本 / テキストへ変換する
    return tokens, predictions  # 返回prompt tokens和预测 / prompt tokenと予測を返す


def generate_texts(tokenizer, model, prompt=PROMPT, num_return_sequences=5, max_new_tokens=20, temperature=1.0, do_sample=True, device=None):  # 生成多个续写 / 複数の続きを生成する
    device = get_device() if device is None else device  # 自动选择设备 / デバイスを自動選択する
    encoding = tokenizer(prompt, return_tensors="pt").to(device)  # 编码prompt / promptを符号化する
    with torch.no_grad():  # 不计算梯度 / 勾配を計算しない
        outputs = model.generate(  # 调用生成API / 生成APIを呼び出す
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]  # 解码生成结果 / 生成結果を復号する


def generate_with_token_probabilities(tokenizer, model, prompt=PROMPT, max_new_tokens=12, device=None):  # 生成并计算每token概率 / 生成し各token確率を計算する
    device = get_device() if device is None else device  # 自动选择设备 / デバイスを自動選択する
    encoding = tokenizer(prompt, return_tensors="pt").to(device)  # 编码prompt / promptを符号化する
    with torch.no_grad():  # 不计算梯度 / 勾配を計算しない
        generated = model.generate(  # 贪心生成 / greedy生成
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_length = encoding["input_ids"].size(1)  # 取得prompt长度 / prompt長を取得する
    new_ids = generated.sequences[0, prompt_length:]  # 取生成token ID / 生成token IDを取る
    rows = []  # 初始化结果 / 結果を初期化する
    for token_id, score in zip(new_ids.tolist(), generated.scores):  # 遍历生成token / 生成tokenを走査する
        probability = torch.softmax(score[0], dim=-1)[token_id].item()  # 计算该token概率 / そのtokenの確率を計算する
        rows.append((tokenizer.decode([token_id]), probability))  # 保存token和概率 / tokenと確率を保存する
    return tokenizer.decode(generated.sequences[0], skip_special_tokens=True), rows  # 返回文本和概率表 / テキストと確率表を返す


def sentence_perplexity(tokenizer, model, sentence, device=None):  # 计算句子perplexity / 文のperplexityを計算する
    device = get_device() if device is None else device  # 自动选择设备 / デバイスを自動選択する
    encoding = tokenizer(sentence, return_tensors="pt").to(device)  # 编码句子 / 文を符号化する
    with torch.no_grad():  # 不计算梯度 / 勾配を計算しない
        loss = model(**encoding, labels=encoding["input_ids"]).loss  # 计算语言模型loss / 言語モデルlossを計算する
    return math.exp(loss.item())  # perplexity = exp(loss) / perplexityはexp(loss)


def build_chat_prompt(messages):  # 构造GPT2用聊天prompt / GPT2用チャットpromptを構築する
    lines = []  # 初始化行列表 / 行リストを初期化する
    for message in messages:  # 遍历消息 / メッセージを走査する
        role = message["role"].capitalize()  # role首字母大写 / roleの先頭を大文字にする
        lines.append(f"{role}: {message['content']}")  # 添加一行 / 1行を追加する
    lines.append("Assistant:")  # 添加助手回答前缀 / アシスタント応答の前置きを追加する
    return "\n".join(lines)  # 返回prompt / promptを返す


def read_sst2_rows(path, max_examples=None):  # 读取SST-2 / SST-2を読む
    rows = []  # 初始化样本列表 / サンプル一覧を初期化する
    with Path(path).open(encoding="utf-8", newline="") as file:  # 打开TSV / TSVを開く
        reader = csv.DictReader(file, delimiter="\t")  # 读取表头 / ヘッダー付きで読む
        for row in reader:  # 遍历行 / 行を走査する
            text = row.get("sentence") or row.get("text")  # 取得文本 / テキストを取得する
            label = row.get("label")  # 取得标签 / ラベルを取得する
            if text is None or label is None:  # 缺少必要列 / 必須列がない場合
                continue  # 跳过 / 飛ばす
            rows.append({"text": text.strip(), "label": int(label)})  # 保存样本 / サンプルを保存する
            if max_examples is not None and len(rows) >= max_examples:  # 达到上限 / 上限に達した場合
                break  # 停止 / 止める
    return rows  # 返回样本 / サンプルを返す


def sentiment_prompt(text):  # 构造感情分析prompt / 感情分析promptを作る
    return f"Review: {text}\nSentiment:"  # 返回prompt / promptを返す


def label_text(label):  # 将标签转换为文本 / ラベルをテキストへ変換する
    return " positive" if int(label) == 1 else " negative"  # 1为positive，0为negative / 1はpositive、0はnegative


def continuation_log_probability(tokenizer, model, prompt, continuation, device=None):  # 计算续写log概率 / 続きのlog確率を計算する
    device = get_device() if device is None else device  # 自动选择设备 / デバイスを自動選択する
    full_text = prompt + continuation  # 合并prompt和续写 / promptと続き結合する
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)  # 编码全文 / 全文を符号化する
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.size(1)  # 计算prompt token数 / prompt token数を計算する
    with torch.no_grad():  # 不计算梯度 / 勾配を計算しない
        logits = model(full_ids).logits[:, :-1, :]  # 取预测下一个token的logit / 次token予測logitを取る
        target = full_ids[:, 1:]  # 目标token / 目標token
        log_probs = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)  # 目标token log概率 / 目標token log確率
    start = max(prompt_len - 1, 0)  # 续写第一个token对应的位置 / 続き最初tokenに対応する位置
    return log_probs[0, start:].sum().item()  # 返回续写log概率总和 / 続きのlog確率合計を返す


def prompt_sentiment_predict(tokenizer, model, text, device=None):  # 用prompt预测情感 / promptで感情を予測する
    prompt = sentiment_prompt(text)  # 构造prompt / promptを作る
    neg_score = continuation_log_probability(tokenizer, model, prompt, " negative", device=device)  # negative得分 / negativeスコア
    pos_score = continuation_log_probability(tokenizer, model, prompt, " positive", device=device)  # positive得分 / positiveスコア
    return 1 if pos_score > neg_score else 0, neg_score, pos_score  # 返回预测和得分 / 予測とスコアを返す


class GPT2EmbeddingDataset(Dataset):  # GPT2埋め込み分類用Dataset / GPT2埋め込み分類用Dataset
    def __init__(self, rows, tokenizer, max_length=128):  # 初始化Dataset / Datasetを初期化する
        self.rows = rows  # 保存样本 / サンプルを保存する
        self.tokenizer = tokenizer  # 保存tokenizer / tokenizerを保存する
        self.max_length = max_length  # 保存最大长度 / 最大長を保存する

    def __len__(self):  # 返回样本数 / サンプル数を返す
        return len(self.rows)  # 返回长度 / 長さを返す

    def __getitem__(self, index):  # 取一个样本 / 1サンプルを取る
        row = self.rows[index]  # 取得行 / 行を取得する
        enc = self.tokenizer(row["text"], truncation=True, max_length=self.max_length)  # 编码文本 / テキストを符号化する
        enc["labels"] = row["label"]  # 添加标签 / ラベルを追加する
        return enc  # 返回样本 / サンプルを返す


def collate_gpt2_classifier(examples, tokenizer):  # 分类任务batch整理 / 分類タスクbatchを整える
    features = [{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in examples]  # 取特征 / 特徴を取る
    batch = tokenizer.pad(features, return_tensors="pt")  # padding / paddingする
    batch["labels"] = torch.tensor([e["labels"] for e in examples], dtype=torch.long)  # 添加标签 / ラベルを追加する
    return batch  # 返回batch / batchを返す


class GPT2EmbeddingClassifier(nn.Module):  # GPT2埋め込み+線形分類器 / GPT2埋め込み+線形分類器
    def __init__(self, encoder, hidden_size):  # 初始化模型 / モデルを初期化する
        super().__init__()  # 调用父类 / 親クラスを呼ぶ
        self.encoder = encoder  # 保存GPT2主体 / GPT2本体を保存する
        self.classifier = nn.Linear(hidden_size, 2)  # 创建线性分类层 / 線形分類層を作る
        for parameter in self.encoder.parameters():  # 遍历GPT2参数 / GPT2パラメータを走査する
            parameter.requires_grad = False  # 固定GPT2 / GPT2を固定する

    def forward(self, input_ids, attention_mask, labels=None):  # 前向计算 / 順伝播を行う
        with torch.no_grad():  # 固定编码器不计算梯度 / 固定エンコーダは勾配なし
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)  # 运行GPT2 / GPT2を実行する
        hidden = outputs.last_hidden_state  # 取得token向量 / tokenベクトルを取得する
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1  # 取得最后有效token位置 / 最後の有効token位置を取得する
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]  # 取最后token向量 / 最後tokenベクトルを取る
        logits = self.classifier(pooled)  # 分类预测 / 分類予測を行う
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None  # 计算loss / lossを計算する
        return loss, logits  # 返回loss和logits / lossとlogitsを返す


def make_lm_rows(rows):  # 构造SFT文本 / SFTテキストを作る
    return [{"text": sentiment_prompt(row["text"]) + label_text(row["label"])} for row in rows]  # prompt加正确标签 / promptに正解ラベルを足す


def make_preference_rows(rows):  # 构造DPO偏好数据 / DPO選好データを作る
    items = []  # 初始化列表 / リストを初期化する
    for row in rows:  # 遍历SST-2样本 / SST-2サンプルを走査する
        correct = label_text(row["label"]).strip()  # 正确标签文本 / 正解ラベルテキスト
        wrong = "negative" if row["label"] == 1 else "positive"  # 错误标签文本 / 不正解ラベルテキスト
        items.append({"prompt": sentiment_prompt(row["text"]), "chosen": f" {correct}", "rejected": f" {wrong}"})  # 添加偏好样本 / 選好サンプルを追加する
    return items  # 返回偏好样本 / 選好サンプルを返す


def to_hf_dataset(rows):  # 转换为Hugging Face Dataset / Hugging Face Datasetへ変換する
    return HFDataset.from_list(rows)  # 从list创建Dataset / listからDatasetを作る
