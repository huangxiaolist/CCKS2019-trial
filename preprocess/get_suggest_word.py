

"""
This .py is to get suggest word in jieba from data/bag_relation_[train|dev|test].txt
& write to file [data/suggest_frep.txt]
"""
fw_suggest_frep = open('data/suggest_frep.txt', 'a')
list_suggest_frep = []
with open('data/bag_relation_train.txt', 'r')as f_train:
    line = f_train.readline()
    while True:
        if not line:
            break
        list = line.strip().split()
        # print(line)

        word1 = list[1]

        if word1 not in list_suggest_frep:
            list_suggest_frep.append(word1)
        word2 = list[2]
        if word2 not in list_suggest_frep:
            list_suggest_frep.append(word2)
        line = f_train.readline()
with open('data/bag_relation_test.txt', 'r') as f_test:
    line = f_test.readline()
    while True:
        if not line:
            break
        list = line.strip().split()

        word1 = list[1]

        if word1 not in list_suggest_frep:
            list_suggest_frep.append(word1)
        word2 = list[2]
        if word2 not in list_suggest_frep:
            list_suggest_frep.append(word2)
        line = f_test.readline()
with open('data/bag_relation_dev.txt', 'r') as f_dev:
    line = f_dev.readline()
    while True:
        if not line:
            break
        list = line.strip().split()

        word1 = list[1]

        if word1 not in list_suggest_frep:
            list_suggest_frep.append(word1)
        word2 = list[2]
        if word2 not in list_suggest_frep:
            list_suggest_frep.append(word2)
        line = f_dev.readline()

total_size = len(list_suggest_frep)
list_suggest_frep_seg1 = list_suggest_frep[0: int(total_size/2)]
list_suggest_frep_seg2 = list_suggest_frep[int(total_size/2): total_size]
str_suggest_frep_seg1 = " ".join(list_suggest_frep_seg1)
str_suggest_frep_seg2 = " ".join(list_suggest_frep_seg2)

fw_suggest_frep.write(str_suggest_frep_seg1)
fw_suggest_frep.write(str_suggest_frep_seg2)
fw_suggest_frep.close()

