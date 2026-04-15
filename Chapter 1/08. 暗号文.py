'与えられた文字列の各文字を、以下の仕様で変換する関数cipherを実装せよ。 英小文字ならば (219 - 文字コード) のASCIIコードに対応する文字に置換 その他の文字はそのまま出力 この関数を用い、英語のメッセージを暗号化・復号化せよ。'
# ======================================
# 08. 暗号文
# ======================================
# 题目：
# 小写字母按219-ASCII转换
#
# 思路：
# ord() + chr()

def cipher(text):
    result = ""
    for c in text:
        if c.islower():                # 小文字かどうかを判定する
            result += chr(219 - ord(c)) # 文字に変換
        else:
            result += c                # 同じように文字列に入る
    return result

text = "Hello World!"
enc = cipher(text)
dec = cipher(enc)

print("08-暗号化:", enc)
print("08-復号化:", dec)
