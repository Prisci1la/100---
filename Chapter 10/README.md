# Chapter 10

10回目用フォルダ。実際の問題範囲は「第9章 80〜84」。

## Scope

- ライブラリ: Hugging Face `transformers`
- モデル: `google-bert/bert-base-uncased`
- 主な目的: BERTのtokenizer/modelを使った第9章前半の課題

## Files

- `knock80.py` - トークン化
- `knock81.py` - マスクの予測
- `knock82.py` - マスクのtop-k予測
- `knock83.py` - CLSトークンによる文ベクトル
- `knock84.py` - 平均による文ベクトル

## Downloads

- 必要なデータセット: なし
- 初回実行時に自動ダウンロードされるもの:
  - `google-bert/bert-base-uncased` のtokenizer
  - `google-bert/bert-base-uncased` のmodel weights

## Local Status

- Python環境: `.venv` の Python 3.12.10
- CUDA: 利用可能
- `transformers`, `datasets`, `evaluate`, `accelerate` は導入済み
- Hugging Face cacheに `google-bert/bert-base-uncased` は取得済み
- 実装時はインターネット接続が必要

## Example

```bash
python knock80.py
python knock81.py
python knock82.py
python knock83.py
python knock84.py
```

モデルは以下のように読み込む。

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
```
