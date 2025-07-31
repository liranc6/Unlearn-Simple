import sys
sys.path.append(".")
sys.path.append("../baselines")

from typing import List, Dict
import torch
from tqdm import tqdm
import zlib
import numpy as np
from sklearn.metrics import auc as get_auc, roc_curve as get_roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os


def compute_ppl(text: str, model, tokenizer, device='cuda'):
    # Tokenize with attention_mask and padding
    inputs = tokenizer(
        text,
        return_tensors='pt',
        add_special_tokens=True
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss, logits = outputs[:2]

    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    input_ids_processed = input_ids[0][1:]
    all_prob = []
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    ppl = torch.exp(loss).item()
    return ppl, all_prob, loss.item()


def inference(text: str, model, tokenizer) -> Dict:
    pred = {}

    _, all_prob, p1_likelihood = compute_ppl(text, model, tokenizer, device=model.device)
    _, _, p_lower_likelihood = compute_ppl(text.lower(), model, tokenizer, device=model.device)
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

    pred["PPL"] = float(p1_likelihood)
    pred["PPL/lower"] = float(p1_likelihood / p_lower_likelihood)
    pred["PPL/zlib"] = float(p1_likelihood / zlib_entropy)

    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min-{int(ratio*100)}%"] = float(-np.mean(topk_prob).item())

    return pred


def eval_data(data: List[str], model, tokenizer):
    out = []
    for text in tqdm(data):
        out.append({'text': text} | inference(text, model, tokenizer))
    return out


def sweep(ppl, y):
    fpr, tpr, _ = get_roc_curve(y, -ppl)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, get_auc(fpr, tpr), acc

def plot_privleak_results(log, auc, plot_dir="privleak_plots"):
    """
    Generates and saves the following plots for privacy leakage evaluation:

    1. Histogram for each metric and split:
       - Shows the distribution of metric values (e.g., PPL, Min-k%) for each data split (forget, retain, holdout).
       - Helps visualize how well the splits are separated by each metric.

    2. ROC curve for each metric and split pair:
       - Plots the True Positive Rate vs. False Positive Rate for distinguishing between two splits using a given metric.
       - The AUC (Area Under Curve) value is shown in the legend.
       - Useful for assessing how well a metric can distinguish between, e.g., forget and retain sets.

    3. AUC heatmap for each metric:
       - Shows a matrix of AUC values for all split pairs (forget, retain, holdout) for a given metric.
       - Provides a quick overview of which splits are most/least separable for each metric.

    Returns:
        plots (dict): Mapping from plot description to saved file path.
    """
    os.makedirs(plot_dir, exist_ok=True)
    plots = {}
    ppl_types = list(log['forget'][0].keys())
    ppl_types.remove('text')
    split_names = ['forget', 'retain', 'holdout']

    # 1. Histograms for each metric and split
    for ppl_type in ppl_types:
        plt.figure()
        for split in split_names:
            values = [d[ppl_type] for d in log[split]]
            sns.histplot(values, kde=True, label=split, stat="density", element="step", fill=True)
        plt.title(f"Histogram: {ppl_type}")
        plt.legend()
        safe_ppl_type = ppl_type.replace("/", "_")
        hist_path = os.path.join(plot_dir, f"hist_{safe_ppl_type}.png")
        plt.savefig(hist_path)
        plt.close()
        plots[f"hist_{safe_ppl_type}"] = plt

    # 2. ROC curves and collect AUCs for heatmap
    auc_matrix = {ppl_type: np.zeros((3, 3)) for ppl_type in ppl_types}
    for i, split0 in enumerate(split_names):
        for j, split1 in enumerate(split_names):
            log0, log1 = log[split0], log[split1]
            for ppl_type in ppl_types:
                auc_key = f"{split0}_{split1}_{ppl_type}"
                auc_score = auc[auc_key]
                auc_matrix[ppl_type][i, j] = auc_score

                # ROC curve
                ppl_nonmember = [d[ppl_type] for d in log0]
                ppl_member = [d[ppl_type] for d in log1]
                ppl = np.array(ppl_nonmember + ppl_member)
                y = np.array([0] * len(ppl_nonmember) + [1] * len(ppl_member))
                fpr, tpr, _ = get_roc_curve(y, -ppl)
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
                plt.plot([0, 1], [0, 1], 'k--', label="Random")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve: {auc_key}")
                plt.legend(loc="lower right")
                safe_auc_key = auc_key.replace("/", "_")
                plot_path = os.path.join(plot_dir, f"roc_{safe_auc_key}.png")
                plt.savefig(plot_path)
                plt.close()
                plots[f"roc_{safe_auc_key}"] = plt

    # 3. AUC Heatmaps for each metric
    for ppl_type in ppl_types:
        plt.figure(figsize=(6, 5))
        sns.heatmap(auc_matrix[ppl_type], annot=True, fmt=".3f", xticklabels=split_names, yticklabels=split_names, cmap="viridis")
        plt.title(f"AUC Heatmap: {ppl_type}")
        safe_ppl_type = ppl_type.replace("/", "_")
        heatmap_path = os.path.join(plot_dir, f"auc_heatmap_{safe_ppl_type}.png")
        plt.savefig(heatmap_path)
        plt.close()
        plots[f"auc_heatmap_{safe_ppl_type}"] = plt

    return plots

def eval(
    forget_data: List[str],
    retain_data: List[str],
    holdout_data: List[str],
    model, tokenizer,
    plot_dir: str = "privleak_plots"
):
    log = {}
    print("Evaluating on the forget set...")
    log['forget'] = eval_data(forget_data, model, tokenizer)
    print("Evaluating on the retain set...")
    log['retain'] = eval_data(retain_data, model, tokenizer)
    print("Evaluating on the holdout set...")
    log['holdout'] = eval_data(holdout_data, model, tokenizer)

    auc = {}
    ppl_types = list(log['forget'][0].keys())
    ppl_types.remove('text')
    for split0 in ['forget', 'retain', 'holdout']:
        for split1 in ['forget', 'retain', 'holdout']:
            log0, log1 = log[split0], log[split1]
            for ppl_type in ppl_types:
                ppl_nonmember = [d[ppl_type] for d in log0]
                ppl_member = [d[ppl_type] for d in log1]
                ppl = np.array(ppl_nonmember + ppl_member)
                y = np.array([0] * len(ppl_nonmember) + [1] * len(ppl_member))
                _, _, auc_score, _ = sweep(ppl, y)
                auc[f"{split0}_{split1}_{ppl_type}"] = auc_score
                
    plots = plot_privleak_results(log, auc, plot_dir=plot_dir)
    
    return auc, log, plots

