import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from utils.metrics_utils import ECELoss, MCELoss, CECELoss, ACELoss


@torch.no_grad()
def model_evaluate(model, loss_fn, data, idx_val, edge_weight=None):
    model.eval()
    output = model(data.x, data.edge_index, edge_weight)
    prob = F.softmax(output, dim=1)
    pred = prob.argmax(dim=-1)

    metrics = {}
    metrics['acc_val'] = int((pred[idx_val] == data.y[idx_val]).sum()) / len(idx_val)
    metrics['loss_val'] = loss_fn(output[idx_val], data.y[idx_val]).cuda()
    metrics['ece_val'] = ECELoss(output[idx_val], data.y[idx_val])

    return metrics


@torch.no_grad()
def model_test(model, loss_fn, data, idx, idx_type="test", edge_weight=None):
    model.eval()

    output = model(data.x, data.edge_index, edge_weight)
    metrics = calculate_all_metrics(output, loss_fn, data, idx, idx_type)
    return metrics


def calculate_all_metrics(output, loss_fn, data, idx, idx_type="test"):
    prob = torch.softmax(output - torch.max(output, dim=1, keepdim=True)[0], dim=1)
    pred = prob.argmax(dim=-1)

    metrics = {}
    metrics['acc_' + idx_type] = (pred[idx] == data.y[idx]).float().mean().item()
    metrics['loss_' + idx_type] = loss_fn(output[idx], data.y[idx]).item()
    metrics['f1_' + idx_type] = f1_score(data.y[idx].cpu().numpy(), pred[idx].cpu().numpy(), average='macro')

    prob = prob.cpu().detach().numpy()

    if np.isnan(prob).any() or np.isinf(prob).any():
        print("Model output contains NaN or infinity.")
        prob = np.nan_to_num(prob)

    try:
        metrics['auc_' + idx_type] = roc_auc_score(data.y[idx].cpu().numpy(), prob[idx], multi_class='ovo')
    except ValueError as e:
        print(f"Error in AUC calculation: {e}")
        metrics['auc_' + idx_type] = 1

    metrics['ece_' + idx_type] = ECELoss(output[idx], data.y[idx])
    metrics['mce_' + idx_type] = MCELoss(output[idx], data.y[idx])
    metrics['cece_' + idx_type] = CECELoss(output[idx], data.y[idx])
    metrics['ace_' + idx_type] = ACELoss(output[idx], data.y[idx])

    return metrics
