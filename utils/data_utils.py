import csv

import pandas as pd
import torch
import numpy as np
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, WebKB, Actor, WikipediaNetwork
from torch.utils.data import random_split
import torch_geometric.transforms as T

from utils.metrics_utils import calculate_mean_and_std


def load_data_from_csv(filename):
    data = pd.read_csv(filename)
    return data


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(dataset_name):
    base_path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        path = osp.join(base_path, 'Planetoid')
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name == 'CoraFull':
        path = osp.join(base_path, 'CoraFull')
        dataset = CoraFull(path, transform=T.NormalizeFeatures())
    elif dataset_name in ['Computers', 'Photo']:
        path = osp.join(base_path, 'Amazon')
        dataset = Amazon(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['CS', 'Physics']:
        path = osp.join(base_path, 'Coauthor')
        dataset = Coauthor(path, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        path = osp.join(base_path, 'WebKB')
        dataset = WebKB(path, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name == 'Actor':
        path = osp.join(base_path, 'Actor')
        dataset = Actor(path, transform=T.NormalizeFeatures())
    elif dataset_name in ['Chameleon', 'Squirrel']:
        path = osp.join(base_path, 'WikipediaNetwork')
        dataset = WikipediaNetwork(path, name=dataset_name, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    return dataset


def get_dataset_idx(args, data):
    train_num = int(data.num_nodes * args.train_ratio)
    val_num = int(data.num_nodes * args.val_ratio)
    test_num = data.num_nodes - train_num - val_num

    idx = range(data.num_nodes)

    train_idx, test_idx = random_split(dataset=idx, lengths=[train_num, val_num + test_num])
    val_idx, test_idx = random_split(dataset=test_idx, lengths=[val_num, test_num])

    return list(train_idx), list(val_idx), list(test_idx)


def save_metrics_to_csv(args, metrics_dict, model_type, filename, mode='a'):
    with open(filename, mode=mode, newline='') as file:
        stats_dict = calculate_mean_and_std(metrics_dict)

        test_metrics = {k: v for k, v in stats_dict.items() if 'test' in k}
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Dataset', 'Model', 'Metric', 'Mean', 'Std Dev', 'train_ratio', 'alpha', 'gamma'])
        for metric, (mean_val, std_dev) in test_metrics.items():
            metric_without_suffix = metric.replace("_test", "")
            writer.writerow(
                [args.dataset, model_type, metric_without_suffix, mean_val, std_dev, args.train_ratio, args.alpha,
                 args.gamma])
