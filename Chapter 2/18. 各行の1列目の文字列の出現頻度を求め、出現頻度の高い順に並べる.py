# q18.py
# 统计第1列出现频率并排序  第1列の出現頻度を数えて並べ替える

from collections import Counter  # 导入计数器  カウンタをインポートする

# 定义输入文件名  入力ファイル名を定義する
filename = r"C:\Priscilla's Vault\100ノック\Chapter 2\popular-names.txt"

# 用列表存储第1列数据  第1列のデータをリストに保存する
names = []

# 读取文件  ファイルを読み込む
with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        # 分割列  列を分割する
        cols = line.strip().split("\t")
        # 添加到列表  リストに追加する
        names.append(cols[0])

# 统计频率  出現頻度を数える
counter = Counter(names)

# 按频率降序排序  出現頻度の降順で並べ替える
result = sorted(counter.items(), key=lambda x: -x[1])

# 输出  結果を出力する
for name, count in result:
    print(count, name)