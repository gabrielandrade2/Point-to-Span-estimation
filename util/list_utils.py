import numpy as np


def flatten_list(l: list):
    return [item for sublist in l for item in sublist]


def list_size(l: list):
    return sum([len(t) for t in l])


def dict_mean(dict_list: list):
    mean_dict = {}
    for key in dict_list[0].keys():
        try:
            mean_dict[key] = np.mean([d[key] for d in dict_list])
            mean_dict[key + '_std'] = np.std([d[key] for d in dict_list])
        except Exception:
            pass
    return mean_dict
