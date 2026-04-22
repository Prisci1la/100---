# q10.py
# 统计文件行数  ファイルの行数を数える

# 定义输入文件名  入力ファイル名を定義する
filename = r"C:\Priscilla's Vault\100ノック\Chapter 2\popular-names.txt"

# 以只读方式打开文件  読み取り専用モードでファイルを開く
with open(filename, "r", encoding="utf-8") as f:
    # 读取所有行到列表中  すべての行をリストとして読み込む
    lines = f.readlines()

# 输出行数（列表长度就是行数）  行数を出力する（リストの長さがそのまま行数になる）
print(len(lines))

#2780