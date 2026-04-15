'“paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を、それぞれ, と として求め、 と の和集合（ ）、積集合（ ）、差集合（ ）を求めよ。さらに、’se’というbi-gramがXおよびYに含まれるかどうかを調べよ。'
# ======================================
# 06. 集合
# ======================================
# 题目：
# 求bi-gram集合并进行集合运算
#
# 思路：
# 用集合推导式生成bi-gram

def bigram(s):
    return {s[i:i+2] for i in range(len(s)-1)}  

X = bigram("paraparaparadise") # 1つ目の文字列
Y = bigram("paragraph")        # 2つ目の文字列

print("06-和集:", X | Y)   # 和集
print("06-積集:", X & Y)   # 積集
print("06-差集:", X - Y)   # 差集

print("06-se in X:", "se" in X)  
print("06-se in Y:", "se" in Y)