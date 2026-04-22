# q15.py
# 将文件分成10个小文件  ファイルを10個の小さいファイルに分割する

import math  # 用于向上取整  切り上げ計算に使用する

# 定义输入文件名  入力ファイル名を定義する
filename = r"C:\Priscilla's Vault\100ノック\Chapter 2\popular-names.txt"

# 读取文件  ファイルを読み込む
with open(filename, "r", encoding="utf-8") as f:
    # 将所有行读取为列表  すべての行をリストとして読み込む
    lines = f.readlines()

# 总行数  総行数
total = len(lines)

# 分成10份  10個に分割する
n = 10

# 每一份的行数（向上取整，避免遗漏）  各ファイルの行数（切り上げて漏れを防ぐ）
chunk_size = math.ceil(total / n)

# 循环分割  ループで分割する
for i in range(n):
    # 计算当前分块的起始和结束位置  現在の分割範囲の開始位置と終了位置を計算する
    start = i * chunk_size
    end = (i + 1) * chunk_size
    
    # 取当前块的数据  現在のブロックのデータを取得する
    chunk = lines[start:end]
    
    # 如果没有数据就停止  データがなければ処理を終了する
    if not chunk:
        break
    
    # 输出文件名  出力ファイル名を作成する
    outname = f"split_{i+1}.txt"
    
    # 写入文件  ファイルに書き込む
    with open(outname, "w", encoding="utf-8") as f:
        f.writelines(chunk)