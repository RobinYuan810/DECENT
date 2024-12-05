import csv
import statistics

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
from torch.autograd import Variable


def initialize_metrics_with_suffixes(metrics, need_suffixes=True):
    if need_suffixes:
        suffixes = ['_test']
        return {f'{metric}{suffix}': [] for metric in metrics for suffix in suffixes}
    else:
        return {f'{metric}': [] for metric in metrics}


def add_metrics(metrics_dict, new_results):
    for metric in metrics_dict:
        if metric in new_results:
            metrics_dict[metric].append(new_results[metric])
        else:
            print(f"Warning: {metric} not found in new_results")


def calculate_mean_and_std(metrics_dict):
    stats_dict = {}
    for metric, values in metrics_dict.items():
        if 'test' in metric:
            if values:
                mean_val = statistics.mean(values)

                std_dev = statistics.stdev(values)

                stats_dict[metric] = (mean_val, std_dev if len(values) > 1 else None)
    return stats_dict


def ECELoss(logits, labels, n_bins=20):
    confidences = F.softmax(logits, dim=1).max(dim=1)[0]
    predictions = torch.argmax(logits, dim=1)
    errors = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = errors[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    if isinstance(ece, torch.Tensor):
        return ece.item()
    else:
        return ece


def MCELoss(logits, labels, n_bins=20):
    confidences = F.softmax(logits, dim=1).max(dim=1)[0]
    predictions = torch.argmax(logits, dim=1)
    errors = predictions.eq(labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    mce = float('-inf')
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        accuracy_in_bin = errors[in_bin].float().mean()
        confidence_in_bin = confidences[in_bin].mean()
        mce = max(mce, torch.abs(accuracy_in_bin - confidence_in_bin))
    if isinstance(mce, torch.Tensor):
        return mce.item()
    else:
        return mce


def CECELoss(logits, labels, n_bins=20):
    num_samples = logits.size(0)
    num_classes = logits.size(1)
    probs = F.softmax(logits, dim=1)
    cece = torch.zeros(1, device=logits.device)
    for class_idx in range(num_classes):
        class_probs = probs[:, class_idx]
        class_labels = (labels == class_idx).float()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        total_in_bin = torch.zeros(n_bins, device=logits.device)
        accuracy_in_bin = torch.zeros(n_bins, device=logits.device)
        confidence_in_bin = torch.zeros(n_bins, device=logits.device)
        for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = (class_probs >= bin_lower) & (class_probs < bin_upper)
            total_in_bin[bin_idx] = in_bin.sum().item()

            if total_in_bin[bin_idx] > 0:
                accuracy_in_bin[bin_idx] = (class_labels[in_bin].sum().item() / total_in_bin[bin_idx])
                confidence_in_bin[bin_idx] = (class_probs[in_bin].mean()).item()
                cece += torch.abs(confidence_in_bin[bin_idx] - accuracy_in_bin[bin_idx]) * (
                        total_in_bin[bin_idx] / num_samples)
    cece /= num_classes
    if isinstance(cece, torch.Tensor):
        return cece.item()
    else:
        return cece


def ACELoss(logits, labels, n_bins=20):
    confidences = F.softmax(logits, dim=1).max(dim=1)[0]
    predictions = torch.argmax(logits, dim=1)
    errors = predictions.eq(labels).float()
    sorted_indices = torch.argsort(confidences)
    equally_spaced_indices = torch.linspace(0, len(confidences) - 1, n_bins + 1).long()

    ace = torch.zeros(1, device=logits.device)
    for i in range(1, n_bins + 1):
        lower = equally_spaced_indices[i - 1]
        upper = equally_spaced_indices[i]
        in_bin = (confidences[sorted_indices].gt(lower.item())) & (confidences[sorted_indices].le(upper.item()))
        if in_bin.any():
            accuracy_in_bin = errors[sorted_indices[in_bin]].float().mean()
            confidence_in_bin = confidences[sorted_indices[in_bin]].mean()
            prop_in_bin = in_bin.float().mean()
            ace += torch.abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin

    if isinstance(ace, torch.Tensor):
        return ace.item()
    else:
        return ace


def calculate_negative_log_likelihood_loss(logits, labels, indices):
    selected_logits = logits[indices]
    selected_labels = labels[indices]
    nll_loss = F.cross_entropy(selected_logits, selected_labels)
    return nll_loss.item()


def calculate_and_print_stats(metrics_dict):
    stats_lines = []
    for metric, values in metrics_dict.items():
        if 'test' in metric:
            if values:
                mean_val = statistics.mean(values)
                if 'auc_test' not in metric:
                    mean_val *= 100
                    stats_line = f"平均{metric}: {mean_val:.2f}"
                else:
                    stats_line = f"平均{metric}: {mean_val:.2f}"
                if len(values) > 1:
                    std_dev = statistics.stdev(values)
                    if 'auc_test' not in metric:
                        std_dev *= 100
                        stats_line += f"±{std_dev:.2f},"
                    else:
                        stats_line += f"±{std_dev:.2f},"
                stats_lines.append(stats_line)
    print(" ".join(stats_lines))


def print_metrics(metrics):
    metrics_str = ', '.join(f'{metric}: {value:.2f}' for metric, value in metrics.items())
    print(metrics_str)
