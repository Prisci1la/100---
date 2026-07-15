# Chapter 14

14回目用フォルダ。実際の問題範囲は「2020版 第10章 94〜99」。

## Scope

- 94: ビーム探索とBLEU曲線
- 95: SentencePieceによるサブワード化
- 96: TensorBoardによる学習過程の可視化
- 97: ハイパーパラメータ調整
- 98: JESC/JParaCrawlなどによるドメイン適応
- 99: Flaskによる翻訳デモサーバ

## Files

- `knock94.py` - ビーム探索
- `knock95.py` - サブワード化
- `knock96.py` - 学習過程の可視化
- `knock97.py` - ハイパー・パラメータの調整
- `knock98.py` - ドメイン適応
- `knock99.py` - 翻訳サーバの構築
- `chapter14_utils.py` - 第14回共用工具

## Downloads

- Chapter 13のKFTTデータと学習済みモデルを基本的に再利用
- 95で必要:
  - SentencePiece modelはKFTTから自前学習予定
- 98で必要になる可能性:
  - JESC
  - JParaCrawl

## Not Downloading Yet

JESC/JParaCrawlはサイズが大きく、どちらを使うかで前処理も変わるため、実装開始時に選んでからダウンロードする。

## Planned Layout

- `checkpoints/`: tuned models
- `data/`: 追加コーパスやsubword済みデータ
- `outputs/`: BLEU曲線、翻訳結果、サーバ出力
- `server/`: Flask demo

## Notes

- BLEU計算は `sacreBLEU`
- 可視化はTensorBoard
- 99はStreamlit/GradioではなくFlaskで作る

## Example

```bash
python knock94.py --beam-sizes 1,2,5,10,20,50,100
python knock95.py --vocab-size 8000
python knock96.py --epochs 5
python knock97.py --batch-sizes 32,64 --lrs 1e-3,1e-4,1e-5,1e-6 --optimizers adam,adamw,radam
python knock98.py --ja-path data/jesc.ja --en-path data/jesc.en
python knock99.py --host 127.0.0.1 --port 5000
```

Chapter 14はChapter 13の `checkpoints/transformer_mt.pt` と前処理済みKFTTを前提にする。
