# q12.py
# 输出末尾10行  末尾10行を出力する

# 定义输入文件名  入力ファイル名を定義する
filename = r"C:\Priscilla's Vault\100ノック\Chapter 2\popular-names.txt"

# 读取文件  ファイルを読み込む
with open(filename, "r", encoding="utf-8") as f:
    # 将所有行读取为列表  すべての行をリストとして読み込む
    lines = f.readlines()

# 取最后10行  最後の10行を取得する
for line in lines[-10:]:
    # end="" 防止重复换行  end=""で改行の重複を防ぐ
    print(line, end="")

'''Liam	M	19837	2018
Noah	M	18267	2018
William	M	14516	2018
James	M	13525	2018
Oliver	M	13389	2018
Benjamin	M	13381	2018
Elijah	M	12886	2018
Lucas	M	12585	2018
Mason	M	12435	2018
Logan	M	12352	2018'''