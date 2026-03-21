import math
import warnings
from collections import Counter

import networkx as nx
import torch
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import average_precision_score as aupr


def pair_aupr(y_pred, y_true):
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy()
    return aupr(y_true.flatten(), y_pred.flatten())


def auc_pytorch(y_pred, y_true, pos_label=1):
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy()
    return auroc(y_pred.flatten(), y_true.flatten(), pos_label)


def auroc(y_pred, y, pos_label):
    fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=pos_label)
    auroc_score = auc(fpr, tpr)
    return auroc_score


def fmax_pytorch(y_pred, y_true, device):
    with torch.no_grad():
        if isinstance(y_pred, list):
            y_pred = [y_.to(device) for y_ in y_pred]
        else:
            y_pred = y_pred.to(device)
        if isinstance(y_true, list):
            y_true = [y_.to(device) for y_ in y_true]
        else:
            y_true = y_true.to(device)

        fmax_, threshold_ = torch.tensor(0.0).to(device), 0.0
        for threshold in (c / 100 for c in range(101)):
            cut = torch.where(y_pred >= threshold, 1, 0)
            correct = cut * y_true
            # Predicted the correct number
            correct_num = correct.sum(dim=1)
            # Avoid error when dividing by 0
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                precision = correct_num / cut.sum(dim=1)
                recall = correct_num / y_true.sum(dim=1)
                avg_precision = torch.mean(precision[torch.bitwise_not(torch.isnan(precision))])
                avg_recall = torch.mean(recall)
            # Skip if the predicted values are all below the threshold
            if torch.isnan(avg_precision) or avg_precision == 0:
                continue
            # Avoid error when dividing by 0
            try:
                fmax_t = 2 * avg_precision * avg_recall / (
                        avg_precision + avg_recall) if avg_precision + avg_recall > 0.0 else 0.0
                if fmax_t > fmax_:
                    fmax_ = fmax_t
                    threshold_ = threshold
            except ZeroDivisionError:
                pass
        return fmax_.cpu().numpy(), threshold_


class SminCalculatorPytorch:
    def __init__(self, go_graph, annots, terms):
        self.go_graph = go_graph
        self.ic = None
        self.terms2idx = terms
        self.idx2terms = {v: k for k, v in self.terms2idx.items()}
        self.calculate_ic(annots)

    def calculate_ic(self, annots):
        cnt = Counter()
        cnt.update(annots)
        self.ic = torch.zeros(len(self.terms2idx))
        for go_id, n in cnt.items():
            if go_id not in self.terms2idx:
                continue
            parents = nx.descendants(self.go_graph, go_id)
            parents = parents.intersection(self.terms2idx.keys())
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
            # self.ic[go_id] = math.log(min_n / n, 2)

            self.ic[self.terms2idx[go_id]] = math.log(min_n / n, 2)

    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.terms2idx:
            raise Exception(f'{go_id} not yet calculated')
        return self.ic[self.terms2idx[go_id]]

    def s_score(self, pred_annots, real_annots):
        total = pred_annots.shape[0]
        # fp is the set of differences between pre and real
        fp = pred_annots - real_annots
        fp = torch.where(fp == 1, 1, 0)
        fp = fp.sum(dim=0)
        mi = (fp * self.ic).sum(dim=0)

        fn = real_annots - pred_annots
        fn = torch.where(fn == 1, 1, 0)
        fn = fn.sum(dim=0)
        ru = (fn * self.ic).sum(dim=0)
        ru /= total
        mi /= total
        s = math.sqrt(ru * ru + mi * mi)
        return s

    def smin_score(self, pred_annots, real_annots, device='cpu'):
        self.ic = self.ic.to(device)
        pred_annots = pred_annots.to(device)
        real_annots = real_annots.to(device)
        smin = 1000
        for t in range(0, 100):
            threshold_ = t / 100.0
            pred_annots_ = torch.where(pred_annots >= threshold_, 1, 0).to(device)

            s = self.s_score(pred_annots_, real_annots)
            if smin > s:
                smin = s
        return smin
