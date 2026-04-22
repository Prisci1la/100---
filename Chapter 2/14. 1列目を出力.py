# q14.py
# 输出前10行的第1列  先頭10行の第1列を出力する

# 定义输入文件名  入力ファイル名を定義する
filename = r"C:\Priscilla's Vault\100ノック\Chapter 2\popular-names.txt"

# 读取文件  ファイルを読み込む
with open(filename, "r", encoding="utf-8") as f:
    # 将所有行读取为列表  すべての行をリストとして読み込む
    lines = f.readlines()

# 遍历前10行  先頭10行を繰り返し処理する
for line in lines[:10]:
    # 去掉换行符并按tab分割  改行を削除してタブで分割する
    cols = line.strip().split("\t")
    
    # 输出第1列（名字）  第1列（名前）を出力する
    print(cols[0])