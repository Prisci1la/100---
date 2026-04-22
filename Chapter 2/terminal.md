# 第2章 UNIXコマンド 実行結果

入力ファイル：popular-names.txt
（名前・性別・人数・年のタブ区切りデータ）

---

## 10. 行数のカウント

**コマンド**

```bash
wc -l popular-names.txt
```

**説明**
ファイルの行数をカウントする。

---

## 11. 先頭から10行を出力

**コマンド**

```bash
head -n 10 popular-names.txt
```

**説明**
ファイルの先頭10行を表示する。

---

## 12. 末尾の10行を出力

**コマンド**

```bash
tail -n 10 popular-names.txt
```

**説明**
ファイルの末尾10行を表示する。

---

## 13. タブをスペースに置換（先頭10行）

**コマンド**

```bash
head -n 10 popular-names.txt | tr '\t' ' '
```

**説明**
タブ文字をスペースに変換して表示する。

---

## 14. 1列目を出力（先頭10行）

**コマンド**

```bash
head -n 10 popular-names.txt | cut -f 1
```

**説明**
各行の1列目（名前）を抽出する。

---

## 15. ファイルを10分割する

**コマンド**

```bash
split -n 10 popular-names.txt split_
```

**説明**
ファイルを10個に分割して保存する。

---

## 16. ランダムに並び替える

**コマンド**

```bash
shuf popular-names.txt
```

（保存する場合）

```bash
shuf popular-names.txt > shuffled.txt
```

**説明**
各行をランダムに並び替える。

---

## 17. 1列目の異なり

**コマンド**

```bash
cut -f 1 popular-names.txt | sort | uniq
```

**説明**
1列目の文字列を重複なく表示する。

---

## 18. 出現頻度（降順）

**コマンド**

```bash
cut -f 1 popular-names.txt | sort | uniq -c | sort -nr
```

**説明**
1列目の出現回数を求め、降順で表示する。

---

## 19. 第3列で降順ソート

**コマンド**

```bash
sort -k 3 -nr popular-names.txt
```

（保存する場合）

```bash
sort -k 3 -nr popular-names.txt > sorted.txt
```

**説明**
第3列（人数）を基準に降順で並び替える。

---

 