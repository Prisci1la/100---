'''“Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.”という文を単語に分解し、
1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字、それ以外の単語は先頭の2文字を取り出し、取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ。'''
# ======================================
# 04. 元素符号
# ======================================
# 题目：
# 指定位置取1个字母，其余取2个字母，生成字典
#
# 思路：
# enumerate获取位置
# 用集合判断是否特殊位置

text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
words = text.replace(".", "").split()   # ピリオドを削除して、単語ごとに分割します
# print(words) # リスト
special = {1, 5, 6, 7, 8, 9, 15, 16, 19} # 特別に1文字だけ取る位置を設定します
result_04 = {} # 辞書型のデータ

for i, word in enumerate(words, 1):      # 自动编号関数 enumerate(words, 1) 1は１から编号する 1から20まで
    if i in special:                     # 特別文字
        key = word[:1]                   # 文字列の先頭から1文字だけ取り出す   [どこから:どこまで]
    else:                                # それ以外
        key = word[:2]                   # 文字列の先頭から2文字だけ取り出す
    result_04[key] = i                         

print("04:", result_04)


# 単語と番号:

# 1: Hi       → H
# 2: He       → He
# 3: Lied     → Li
# 4: Because  → Be
# 5: Boron    → B
# 6: Could    → C
# 7: Not      → N
# 8: Oxidize  → O
# 9: Fluorine → F
# ...

# ↓ 辞書

# {
#  H:1,
#  He:2,
#  Li:3,
#  Be:4,
#  B:5,
#  ...
# }