import jieba


def outofdate_func():
    """
    Set suggest word in jieba & write the sentence word-cut to data/text.segment.txt
    :return:
    """
    list_suggest_word = []
    with open('data/suggest_frep.txt') as fr:
        line = fr.readline()
        while True:
            if not line:
                break
            lists = line.strip().split()
            list_suggest_word = list_suggest_word + lists
            line = fr.readline()
    for suggest_word in list_suggest_word:
        jieba.suggest_freq(suggest_word, True)

    with open('data/text.txt', 'r') as f:
        fw = open('data/text_segment.txt', 'w')
        while True:

            line = f.readline()
            if not line:
                line = f.readline()
                if not line:
                    break

            seg_list = jieba.cut(line, cut_all=False)

            new_line = " ".join(seg_list)

            # print(new_line)
            fw.write(new_line)
        fw.close()
    print('the cut processor done!')


# def coverageSeament():
#     with open('data/text_cut.txt', 'a') as fw:
#         i = 0
#         while i < 10:
#             with open('data/text_segment_'+str(i)+'.txt', 'r') as f:
#                 line = f.readline()
#                 while True:
#                     if not line:
#                         line = f.readline()
#                         if not line:
#                             i += 1
#                             break
#                     fw.write(line)


if __name__ == '__main__':
    # coverageSeament()
    outofdate_func()