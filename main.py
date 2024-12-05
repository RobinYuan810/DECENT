import os.path

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sipbuild.generator.parser.annotations import boolean
from torch_geometric.utils import index_to_mask
import argparse

from models.base_model import base_main
from models.calib_model import original_calib_main, calib_main, calculate_edge_weights
from utils.data_utils import load_data_from_csv, get_dataset_idx, load_dataset, set_seed, save_metrics_to_csv
from utils.metrics_utils import initialize_metrics_with_suffixes, add_metrics
from models.model_gnn import Temperature_Scalling, VS, CaGCN, GATS, initialize_model_and_optimizer
from utils.plotting_utils import plot_calib_perform_diagrams


def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default="DECENT",
                        help='record modelname, mainly used to show current results explictly.'
                             'value: DECENT, DECENT-WithoutNTM, DECENT-WithoutETM, DECENT-WithoutNTM-alpha, DECENT-WithoutNTM-gamma, DECENT-WithoutETM-homo')
    parser.add_argument('--contrastname', type=str, default="Original GNN",
                        help="base_model calib_model 1~8")
    parser.add_argument('--backbone', type=str, default="GCN")
    parser.add_argument('--dataset', type=str, default="PubMed",
                        help='dataset for training')
    parser.add_argument('--train_ratio', type=float, default=0.60)
    parser.add_argument('--val_ratio', type=float, default=0.05)

    parser.add_argument('--loss_function', type=str, default="cross_entropy")
    parser.add_argument('--gpu_device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden_channels.')
    parser.add_argument('--dropout', type=float, default=0.7,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--if_draw', type=str, default="yes", help="control if draw results pics or not")
    parser.add_argument('--base_num', type=int, default=20,
                        help='calculate the basic results multiple times to obtain stable results')
    parser.add_argument('--base_epochs', type=int, default=1000,
                        help='Number of calib_epochs to train base_model')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='base_model learning rate.')
    parser.add_argument('--base_weight_decay', type=float, default=5e-4,
                        help='base_model weight decay (L2 loss on parameters) reduce overfitting.')
    parser.add_argument('--calib_num', type=int, default=5,
                        help='calculate the calib results multiple times to obtain stable results')
    parser.add_argument('--calib_epochs', type=int, default=1000,
                        help='Number of calib_epochs to train calib_model.')
    parser.add_argument('--calib_lr', type=float, default=0.01)
    parser.add_argument('--calib_weight_decay', type=float, default=5e-3,
                        help='Weight decay (L2 loss on parameters) for calibration.')
    parser.add_argument('--early_stopping_patience', type=int, default=100,
                        help="Early termination to prevent overfitting")

    parser.add_argument('--n_bins', type=int, default=20,
                        help='calib metrics such as ece use it to decide number of bins')

    parser.add_argument('--use_etm', type=str, default="yes", help="use etm to adjust edge_weight")
    parser.add_argument('--ew_method', type=str, default="MLP", help="method to calculate edge_weight")
    parser.add_argument('--use_etm_homo', type=str, default="yes",
                        help="etm's homophilic edges after decisive computing")
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--coefficient_method', type=str, default="cosine", help="such as euclidean, cosine")
    parser.add_argument('--epsilon', type=float, default=0.5, help="Smooth the difference to prevent dividing by zero")
    parser.add_argument('--use_ntm', type=str, default="yes", help="use ntm to design loss func for node")
    parser.add_argument('--alpha', type=float, default=0.09, help="NTM entropy")
    parser.add_argument('--gamma', type=float, default=0.01, help="NTM logit1/2")
    return parser.parse_args()


def main():
    base_results = initialize_metrics_with_suffixes(metrics)
    calib_results = initialize_metrics_with_suffixes(metrics)

    combinations = ["", "DECENT"]
    original_calib_methods = ["VS", "TS", "CaGCN", "GATS"]
    name_list = []
    for method in original_calib_methods:
        for comb in combinations:
            if comb == "DECENT":
                name_list.append(f"{comb}+{method}")
            else:
                name_list.append(f"{method}")
    contrast_results = {f'{name}': initialize_metrics_with_suffixes(metrics) for name in name_list}
    for i in range(args.base_num):
        idx_train, idx_val, idx_test = get_dataset_idx(args, data)

        base_model, optimizer_base = initialize_model_and_optimizer(args, nfeat, nclass, data, device)
        results = base_main(args, base_model, optimizer_base, data, idx_train, idx_val, idx_test, i)
        add_metrics(base_results, results)

        for j in range(args.calib_num):

            save_root = f'results/{args.modelname}/{args.backbone}/{args.dataset}/saved_models/'
            save_name = f'{args.contrastname}.pth'
            save_path = save_root + save_name
            state_dict = torch.load(save_path)
            base_model.load_state_dict(state_dict, strict=False)
            for param in base_model.parameters():
                param.requires_grad = True

            args.contrastname = "DECENT(ours)"
            edge_weight = None
            if args.use_etm == "yes":
                edge_weight = calculate_edge_weights(args, base_model, nclass, data, device, idx_train, idx_val,
                                                     idx_test)
            base_model.load_state_dict(state_dict, strict=False)
            for param in base_model.parameters():
                param.requires_grad = True
            results = calib_main(args, base_model, optimizer_base, data, idx_train, idx_val, idx_test, edge_weight, j)
            add_metrics(calib_results, results)

            base_model.load_state_dict(state_dict, strict=False)
            ts = Temperature_Scalling(base_model).to(device)
            optimizer_ts = optim.Adam(filter(lambda p: p.requires_grad, ts.parameters()),
                                      lr=args.calib_lr, weight_decay=args.calib_weight_decay)
            args.contrastname = "TS"
            results = original_calib_main(args, ts, optimizer_ts, data, idx_train, idx_val, idx_test, j)
            add_metrics(contrast_results[f'{args.contrastname}'], results)

            args.contrastname = "DECENT+TS"
            results = calib_main(args, ts, optimizer_ts, data, idx_train, idx_val, idx_test, edge_weight, j)
            add_metrics(contrast_results[f'{args.contrastname}'], results)
            base_model.load_state_dict(state_dict, strict=False)
            vs = VS(base_model, nclass).to(device)
            optimizer_vs = optim.Adam(filter(lambda p: p.requires_grad, vs.parameters()),
                                      lr=args.calib_lr, weight_decay=args.calib_weight_decay)
            args.contrastname = "VS"
            results = original_calib_main(args, vs, optimizer_vs, data, idx_train, idx_val, idx_test, j)
            add_metrics(contrast_results[f'{args.contrastname}'], results)

            args.contrastname = "DECENT+VS"
            results = calib_main(args, vs, optimizer_vs, data, idx_train, idx_val, idx_test, edge_weight, j)
            add_metrics(contrast_results[f'{args.contrastname}'], results)
            base_model.load_state_dict(state_dict, strict=False)
            cagcn = CaGCN(base_model, nclass, args.hidden).to(device)
            optimizer_cagcn = optim.Adam(filter(lambda p: p.requires_grad, cagcn.parameters()),
                                         lr=args.calib_lr, weight_decay=args.calib_weight_decay)
            args.contrastname = "CaGCN"
            results = original_calib_main(args, cagcn, optimizer_cagcn, data, idx_train, idx_val, idx_test, j)
            add_metrics(contrast_results[f'{args.contrastname}'], results)

            args.contrastname = "DECENT+CaGCN"
            results = calib_main(args, cagcn, optimizer_cagcn, data, idx_train, idx_val, idx_test, edge_weight, j)
            add_metrics(contrast_results[f'{args.contrastname}'], results)
            base_model.load_state_dict(state_dict, strict=False)
            train_mask = index_to_mask(torch.LongTensor(idx_train), data.num_nodes).cuda()
            gats = GATS(base_model, data.edge_index, data.num_nodes, train_mask, dataset.num_classes)
            optimizer_gats = optim.Adam(filter(lambda p: p.requires_grad, gats.parameters()),
                                        lr=args.calib_lr, weight_decay=args.calib_weight_decay)
            args.contrastname = "GATS"
            results = original_calib_main(args, gats, optimizer_gats, data, idx_train, idx_val, idx_test, j)
            add_metrics(contrast_results[f'{args.contrastname}'], results)

            args.contrastname = "DECENT+GATS"
            results = calib_main(args, gats, optimizer_gats, data, idx_train, idx_val, idx_test, edge_weight, j)
            add_metrics(contrast_results[f'{args.contrastname}'], results)

    filename = f'results/{args.modelname}/{args.backbone}/{args.dataset}/ret.csv'
    save_metrics_to_csv(args, base_results, 'base', filename, mode='a')

    save_metrics_to_csv(args, calib_results, 'DECENT', filename, mode='a')

    for model_type, metrics_dict in contrast_results.items():
        save_metrics_to_csv(args, metrics_dict, model_type, filename, mode='a')


if __name__ == '__main__':
    args = get_default_args()

    set_seed(args.seed)
    torch.cuda.set_device(args.gpu_device)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = load_dataset(args.dataset)

    data = dataset[0].to(device)
    nclass = data.y.max().item() + 1
    nfeat = data.x.shape[1]

    perform_metrics = ['acc', 'auc', 'f1']
    calib_metrics = ['ece', 'ace', 'mce', 'cece']
    metrics = perform_metrics + calib_metrics
    print(
        f"训练开始：模型名为{args.modelname}，backbone为{args.backbone}，数据集为{args.dataset}，train_ratio是{args.train_ratio}，gpu是{args.gpu_device}，"
        f"use_ntm是{args.use_ntm}，alpha是{args.alpha}，gamma是{args.gamma}，use_etm是{args.use_etm}，use_etm_homo是{args.use_etm_homo}")

    main()
    if args.if_draw == 'yes':
        filename = f'results/{args.modelname}/{args.backbone}/{args.dataset}/ret.csv'
        ret = load_data_from_csv(filename)
        perform_metrics = ['acc', 'auc']
        calib_metrics = ['ece', 'ace']
        for perform_metric in perform_metrics:
            for calib_metric in calib_metrics:
                plot_calib_perform_diagrams(args, ret, calib_metric, perform_metric)
