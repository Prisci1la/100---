# q16.py
# 随机打乱文件行  ファイルの行をランダムに並び替える

import random  # 导入随机模块  乱数モジュールをインポートする

# 定义输入文件名  入力ファイル名を定義する
filename = r"C:\Priscilla's Vault\100ノック\Chapter 2\popular-names.txt"

# 读取文件  ファイルを読み込む
with open(filename, "r", encoding="utf-8") as f:
    # 将所有行读取为列表  すべての行をリストとして読み込む
    lines = f.readlines()

# 打乱列表顺序  リストの順序をランダムに並び替える
random.shuffle(lines)

# 输出到新文件  新しいファイルに出力する
with open("shuffled.txt", "w", encoding="utf-8") as f:
    f.writelines(lines)