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

'''Mary	F	7065	1880
Anna	F	2604	1880
Emma	F	2003	1880
Elizabeth	F	1939	1880
Minnie	F	1746	1880
Margaret	F	1578	1880
Ida	F	1472	1880
Alice	F	1414	1880
Bertha	F	1320	1880
Sarah	F	1288	1880'''