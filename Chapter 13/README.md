# Chapter 13

13回目用フォルダ。実際の問題範囲は「2020版 第10章 90〜93」。

## Scope

- 90: KFTTデータの準備
- 91: PyTorchだけでTransformer機械翻訳モデルを訓練
- 92: 学習済みモデルで日本語文を英語へ翻訳
- 93: `sacreBLEU` でBLEUスコアを計測

## Files

- `knock90.py` - データの準備
- `knock91.py` - 機械翻訳モデルの訓練
- `knock92.py` - 機械翻訳モデルの適用
- `knock93.py` - BLEUスコアの計測
- `chapter13_utils.py` - 第13回共用工具

## Downloads

- 必要:
  - KFTT: `https://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz`
- まだ存在しない:
  - `data/kftt-data-1.0.tar.gz`
  - `data/kftt-data-1.0/`

## Planned Layout

- `data/raw/`: ダウンロードした元データ
- `data/processed/`: 前処理後のtrain/dev/test
- `checkpoints/`: 学習済みTransformer
- `outputs/`: 翻訳結果とBLEU結果

## Notes

- OpenNMTやfairseqは使用しない
- PyTorchの `torch.nn.Transformer` は使用可
- 学習はGPUとbatch trainingを前提にする
- 日本語は形態素、英語は単語単位を基本にする

## Example

```bash
python knock90.py
python knock91.py --epochs 5 --batch-size 64
python knock92.py --sentence "京都 は 美しい 。"
python knock93.py --split test
```

短い確認だけ行う場合:

```bash
python knock90.py --max-lines 1000 --vocab-size 5000 --min-freq 1
python knock91.py --epochs 1 --batch-size 16 --emb-size 128 --nhead 4 --num-layers 2 --max-train-examples 500
python knock93.py --split dev --max-examples 50
```
