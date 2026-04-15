#“Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.”という文を単語に分解し、各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ。
# ======================================
# 03. 圆周率
# ======================================
# 题目：
# 分词并统计每个单词长度
#
# 思路：
# 用正则提取字母单词
# 用len计算长度
import re
text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
words = re.findall("[A-Za-z]+", text)   # 
# words = re.findall("[A-Za-z]", text)
result_03 = [len(word) for word in words] # 文字数を出す
print("03:", result_03)

# re.findall(pattern, string)
# "[A-Za-z]+" A-Za-zはすべでの大小文字のアルファベット　+　は1回以上続きます

# Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics
#  3  1  4   1   5        9     2    6       5    3    5      8        9         7       9