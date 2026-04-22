# q13.py
# 将前10行中的tab替换为空格  先頭10行のタブをスペースに置換する

# 定义输入文件名  入力ファイル名を定義する
filename = r"C:\Priscilla's Vault\100ノック\Chapter 2\popular-names.txt"

# 读取文件  ファイルを読み込む
with open(filename, "r", encoding="utf-8") as f:
    # 将所有行读取为列表  すべての行をリストとして読み込む
    lines = f.readlines()

# 遍历前10行  先頭10行を繰り返し処理する
for line in lines[:10]:
    # 将\t替换为空格  \tをスペースに置換する
    new_line = line.replace("\t", " ")
    # 输出结果  結果を出力する
    print(new_line, end="")