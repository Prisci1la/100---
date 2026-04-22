# q19.py
# 按第3列数值降序排序  第3列の数値で降順に並べ替える

# 定义输入文件名  入力ファイル名を定義する
filename = r"C:\Priscilla's Vault\100ノック\Chapter 2\popular-names.txt"

# 读取文件  ファイルを読み込む
with open(filename, "r", encoding="utf-8") as f:
    # 将所有行读取为列表  すべての行をリストとして読み込む
    lines = f.readlines()

# 定义排序规则函数  並べ替えルールの関数を定義する
def get_third_col(line):
    # 分割列  列を分割する
    cols = line.strip().split("\t")
    
    # 返回第3列（人数），转成整数  第3列（人数）を整数に変換して返す
    return int(cols[2])

# 排序（降序）  降順で並べ替える
sorted_lines = sorted(lines, key=get_third_col, reverse=True)

# 输出到文件  ファイルに出力する
with open("sorted.txt", "w", encoding="utf-8") as f:
    f.writelines(sorted_lines)