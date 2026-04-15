'''スペースで区切られた単語列に対して、各単語の先頭と末尾の文字は残し、それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ。ただし、長さが４以下の単語は並び替えないこととする。
適当な英語の文（例えば”I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .”）を与え、その実行結果を確認せよ。'''
# ======================================
# 09. Typoglycemia
# ======================================
# 题目：
# 打乱单词中间部分
#
# 思路：
# 中间转list后shuffle
import random
def shuffle_word(word):
    if len(word) <= 4:
        return word

    middle = list(word[1:-1])         #  0 1 2 3 4 5
    random.shuffle(middle)            # shuffle
    return word[0] + "".join(middle) + word[-1] 

text = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
words = text.split()
print(words)
result_09 = " ".join(shuffle_word(w) for w in words)
print("09:", result_09)