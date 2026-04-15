'文字列”stressed”の文字を逆に（末尾から先頭に向かって）並べた文字列を得よ。'
# ======================================
# 02. 字符串逆序
# ======================================
# 题目：
# 反转 "stressed"
#
# 思路：
# 用切片[::-1]

s = "stressed"
result_02 = ""                     
for i in range(len(s)-1, -1, -1):  # range(start, stop, step)
    result_02 = result_02 + s[i]   #start=len（）-1 最後から　stop=-1 最後まで　step=-1 逆方向に進む
print("02:", result_02)

# d
# de
# des
# dess
# desse
# desser
# dessert
# desserts