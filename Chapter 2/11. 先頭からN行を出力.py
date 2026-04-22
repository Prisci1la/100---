# q11.py
# 输出前10行  先頭10行を出力する

# 定义输入文件名  入力ファイル名を定義する
filename = r"C:\Priscilla's Vault\100ノック\Chapter 2\popular-names.txt"

# 读取文件  ファイルを読み込む
with open(filename, "r", encoding="utf-8") as f:
    # 将所有行读取为列表  すべての行をリストとして読み込む
    lines = f.readlines()

# 切片取前10行  スライスで先頭10行を取得する
for line in lines[:10]:
    # end="" 防止重复换行  end=""で改行の重複を防ぐ
    print(line, end="")