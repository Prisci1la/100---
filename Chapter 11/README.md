# Chapter 11

11回目用フォルダ。実際の問題範囲は「第9章 85〜89」。

## Scope

- ライブラリ: Hugging Face `transformers`
- モデル: `google-bert/bert-base-uncased`
- 主な目的: BERTのファインチューニングと評価

## Files

- `knock85.py` - データセットの準備
- `knock86.py` - ミニバッチの作成
- `knock87.py` - ファインチューニング
- `knock88.py` - 極性分析
- `knock89.py` - アーキテクチャの変更
- `chapter11_utils.py` - 第11回共用工具

## Downloads

- 新しく必要なデータセット: なし
- 既存データを利用:
  - `../Chapter 8/data/SST-2/train.tsv`
  - `../Chapter 8/data/SST-2/dev.tsv`
- 初回実行時に自動ダウンロードされるもの:
  - `google-bert/bert-base-uncased` のtokenizer/model weights

## Training Rule

- 最適化・保存対象のスコアは「最後のepoch」ではなく、全epoch中のbest valid accuracyまたはbest valid loss
- 学習率の目安: `1e-3`〜`1e-6`
- Optimizer候補: `Adam`, `AdamW`
- `transformers.Trainer` を使っても、素のPyTorchで書いてもよい

## Local Status

- SST-2は既に存在する
- Hugging Face cacheに `google-bert/bert-base-uncased` は取得済み
- `.venv` に `transformers`, `datasets`, `evaluate`, `accelerate` は導入済み

## Example

```bash
python knock85.py
python knock86.py
python knock87.py --epochs 3 --batch-size 16 --lr 2e-5
python knock88.py
python knock89.py --epochs 3 --batch-size 16 --lr 2e-5
```

短い確認だけ行う場合:

```bash
python knock87.py --epochs 1 --max-train-examples 32 --max-dev-examples 32 --batch-size 8
python knock89.py --epochs 1 --max-train-examples 32 --max-dev-examples 32 --batch-size 8
```
