import os

import torch
import torch.nn.functional as F
from models.eval_test import model_evaluate, model_test
from utils.loss_function_utils import get_loss_function
from utils.plotting_utils import plot_reliability_diagrams, plot_histograms


def base_train(model, optimizer, loss_fn, data, idx_train, edge_weight=None):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, edge_weight)
    loss = loss_fn(output[idx_train], data.y[idx_train]).cuda()
    loss.backward()
    optimizer.step()


def base_main(args, model, optimizer_t, data, idx_train, idx_val, idx_test, i=-1):
    best_model_state = None
    best = 0
    for epoch in range(args.base_epochs):
        loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None,
                                            reduction='mean')
        base_train(model, optimizer_t, loss_fn, data, idx_train)
        metrics = model_evaluate(model, loss_fn, data, idx_val)

        if metrics['acc_val'] > best:
            best = metrics['acc_val']
            best_model_state = model.state_dict()

    save_root = f'results/{args.modelname}/{args.backbone}/{args.dataset}/saved_models/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_name = f'{args.contrastname}.pth'
    save_path = save_root + save_name
    torch.save(best_model_state, save_path)
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)
    output = model(data.x, data.edge_index)
    if args.if_draw == 'yes':
        if i == 0 or i == args.base_num - 1:
            plot_histograms(args, idx_train, output, data.y, "train_confidence density")
            plot_histograms(args, idx_test, output, data.y)
            plot_reliability_diagrams(args, idx_test, output, data.y)

    results = model_test(model, loss_fn, data, idx_test, "test")
    return results
