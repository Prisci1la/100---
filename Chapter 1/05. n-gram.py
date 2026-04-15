'与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ。この関数を用い、”I am an NLPer”という文から文字tri-gram、単語bi-gramを得よ。'
# ======================================
# 05. n-gram
# ======================================
# 题目：
# 实现n-gram函数
#
# 思路：
# 滑动窗口切片

def ngram(seq, n):
    result = []
    for i in range(len(seq) - n + 1):  # 控制窗口范围
        result.append(seq[i:i+n])     # 取连续n个元素
    return result

text = "I am an NLPer"

char_tri = ngram(text, 3)          # 文字3-gram
word_bi = ngram(text.split(), 2)   # 単語2-gram

print("05-char:", char_tri)
print("05-word:", word_bi)

# I am an NLPer
#'I a'
# ' am'
#  'am '
#   'm a'
#    ' an'
#     'an '
#      'n N'
#       ' NL'
#        'NLP'
#         'LPe'
#          'Per'

# I am an NLPer
#'I am'
#  'am an'
#     'an NLPer'