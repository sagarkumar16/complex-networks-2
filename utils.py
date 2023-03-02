from collections import defaultdict


def count_faster(l):
    dic = {}

    for i in l:

        if i in dic.keys():
            dic[i] += 1

        else:
            dic[i] = 1

    return [(key, dic[key]) for key in dic.keys()]
