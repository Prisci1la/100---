# q17.py
# 获取第1列的不同值  第1列の異なる値を取得する

# 定义输入文件名  入力ファイル名を定義する
filename = r"C:\Priscilla's Vault\100ノック\Chapter 2\popular-names.txt"

# 使用集合存储唯一值（自动去重）  集合を使って一意な値を保存する（自動で重複を除去）
names = set()

# 读取文件  ファイルを読み込む
with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        # 分割列  列を分割する
        cols = line.strip().split("\t")
        
        # 加入集合  集合に追加する
        names.add(cols[0])

# 排序后输出（更美观）  ソートして出力する（見やすくするため）
for name in sorted(names):
    print(name)

'''Abigail
Aiden
Alexander
Alexis
Alice
Amanda
Amelia
Amy
Andrew
Angela
Anna
Annie
Anthony
Ashley
Austin
Ava
Barbara
Benjamin
Bertha
Bessie
Betty
Brandon
Brian
Brittany
Carol
Carolyn
Charles
Charlotte
Chloe
Christopher
Clara
Crystal
Cynthia
Daniel
David
Deborah
Debra
Donald
Donna
Doris
Dorothy
Edward
Elijah
Elizabeth
Emily
Emma
Ethan
Ethel
Evelyn
Florence
Frances
Frank
Gary
George
Hannah
Harper
Harry
Heather
Helen
Henry
Ida
Isabella
Jacob
James
Jason
Jayden
Jeffrey
Jennifer
Jessica
Joan
John
Joseph
Joshua
Judith
Julie
Justin
Karen
Kathleen
Kelly
Kimberly
Larry
Laura
Lauren
Liam
Lillian
Linda
Lisa
Logan
Lori
Lucas
Madison
Margaret
Marie
Mark
Mary
Mason
Matthew
Megan
Melissa
Mia
Michael
Michelle
Mildred
Minnie
Nancy
Nicholas
Nicole
Noah
Oliver
Olivia
Pamela
Patricia
Rachel
Rebecca
Richard
Robert
Ronald
Ruth
Samantha
Sandra
Sarah
Scott
Sharon
Shirley
Sophia
Stephanie
Steven
Susan
Tammy
Taylor
Thomas
Tracy
Tyler
Virginia
Walter
William'''