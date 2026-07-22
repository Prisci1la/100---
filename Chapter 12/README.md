# Chapter 12

12回目用フォルダ。実際の問題範囲は「第10章」。

## Scope

- モデル: `openai-community/gpt2-medium`
- 選好チューニングライブラリ: Hugging Face `trl`
- `knock98.py` と `knock99.py` は既定で4bit量子化ロード + LoRAで学習する

## Files

- `knock90.py` - 次単語予測
- `knock91.py` - 続きのテキストの予測
- `knock92.py` - 予測されたテキストの確率を計算
- `knock93.py` - パープレキシティ
- `knock94.py` - チャットテンプレート
- `knock95.py` - マルチターンのチャット
- `knock96.py` - プロンプトによる感情分析
- `knock97.py` - 埋め込みに基づく感情分析
- `knock98.py` - ファインチューニング
- `knock99.py` - 選好チューニング
- `chapter12_utils.py` - 第12回共用工具

## Downloads

- 既存データを利用:
  - `../Chapter 8/data/SST-2/train.tsv`
  - `../Chapter 8/data/SST-2/dev.tsv`
- 初回実行時に自動ダウンロードされるもの:
  - `openai-community/gpt2-medium` のtokenizer
  - `openai-community/gpt2-medium` のmodel weights

## Local Status

- Hugging Face cacheに `openai-community/gpt2-medium` があれば再利用される
- `gpt2-medium` は約355M parametersなので、BERTよりGPUメモリと保存容量を多めに使う
- `.venv` に `trl` と `peft` は導入済み
- 4bit量子化には `bitsandbytes` が必要
- 実行時はモデル初回取得のためインターネット接続が必要

## Full Example

```bash
python knock90.py
python knock91.py
python knock92.py
python knock93.py
python knock94.py
python knock95.py
python knock96.py
python knock97.py --epochs 3
python knock98.py --epochs 1 --batch-size 2 --gradient-accumulation-steps 8
python knock99.py --epochs 1 --batch-size 1 --gradient-accumulation-steps 8
```

軽量モデルで構文に近い動作だけ試す場合は `--model-name distilgpt2` を指定する。
`knock98.py` と `knock99.py` は既定で全train splitを使うため、短い確認だけ行う場合は `--max-train-examples` を指定する。
4bitを使わずに通常ロードする場合は `--no-4bit` を指定する。

## Quick Check

```bash
python knock96.py --max-examples 100
python knock97.py --epochs 1 --max-train-examples 1000 --max-dev-examples 200
python knock98.py --max-train-examples 256
python knock99.py --max-train-examples 128
```

## Notes

基本の読み込み予定。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
```

4bit LoRA学習の既定設定。

```bash
python knock98.py --epochs 1 --batch-size 2 --gradient-accumulation-steps 8
python knock99.py --epochs 1 --batch-size 1 --gradient-accumulation-steps 8
```

## Multi-GPU on AIX

`knock98.py` と `knock99.py` は `torchrun` で複数GPUに分散できる。

```bash
torchrun --nproc_per_node=8 knock98.py --epochs 1 --batch-size 2 --gradient-accumulation-steps 1
torchrun --nproc_per_node=8 knock99.py --epochs 1 --batch-size 1 --gradient-accumulation-steps 1
```

4bit量子化ロード時は各プロセスが `LOCAL_RANK` のGPUを使う。

## Legacy Torch on AIX

AIXの標準環境が `torch==1.5.1` の場合、`peft` / `trl` / `bitsandbytes` は互換性問題で動かないことがある。
`knock98.py` と `knock99.py` は `--legacy-torch auto` が既定なので、古いtorchでは手書きのPyTorch training loopへ自動的に切り替える。

このfallbackでは4bit LoRAを使わず、既定ではGPT-2-mediumの最後のTransformer blockとLM headだけを学習する。

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env knock98.py --epochs 1 --batch-size 2 --gradient-accumulation-steps 8
python -m torch.distributed.launch --nproc_per_node=8 --use_env knock99.py --epochs 1 --batch-size 1 --gradient-accumulation-steps 8
```

軽く確認する場合。

```bash
python knock98.py --max-train-examples 128 --batch-size 2
python knock99.py --max-train-examples 64 --batch-size 1
```

`--trainable all` は全パラメータを更新するが、AIX標準環境ではメモリと時間が大きく増える。
