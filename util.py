from collections import defaultdict


def infinite_dict():
    return defaultdict(infinite_dict)

