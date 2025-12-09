#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module provides clustering functions, to group different transcriptions via their pairwise alignment cost into clusters associated to different structural realisations TODO.
"""

import numpy as np

from itertools import product
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
)


def groups_to_labels(true_groups, predicted_groups):
    """
    Convert group dictionaries to aligned label arrays.

    Parameters
    ----------
    true_groups: dict like {group_name: [item1, item2, ...]}
    predicted_groups: dict like {group_name: [item1, item2, ...]}

    Example:
    true_test = {'group_1': [134, 145, 167], 'group_2': [124, 125], 'group_3': [155, 156]}
    pred_test = {'group_1': [124, 125], 'group_2': [134, 145, 167], 'group_3': [155], 'group_4': [156]}

    Returns two arrays of the same length with labels for each item
    true labels: [1 1 0 0 2 2 0]
    pred_labels: [0 0 1 1 2 3 1]

    """
    # Get all items from TRUE groups (use true groups as reference)
    all_items = set()
    for group in true_groups.values():
        all_items.update(group)

    # sort all items and map to index
    all_items = sorted(all_items)
    item_to_idx = {item: i for i, item in enumerate(all_items)}

    # Create label arrays
    n_items = len(all_items)
    true_labels = np.full(n_items, -1, dtype=int)
    pred_labels = np.full(n_items, -1, dtype=int)

    # assign group index to each item according to true grouping
    for group_idx, (_, members) in enumerate(true_groups.items()):
        for item in members:
            if item in item_to_idx:
                true_labels[item_to_idx[item]] = group_idx

    # same for predictions
    for group_idx, (_, members) in enumerate(predicted_groups.items()):
        for item in members:
            if item in item_to_idx:
                pred_labels[item_to_idx[item]] = group_idx

    return true_labels, pred_labels, all_items


def compute_pairwise_f1(true_labels, pred_labels):
    """
    Compute F1 score based on pairwise same-cluster decisions.

    For each pair of items check: should they be in the same cluster?

    Parameters
    ----------
    true_labels: np.ndarray
        Array of all group items with their reference group labels
    pred_labels: np.ndarray
        Array of all group items with their predicted group labels

    """
    n = len(true_labels)

    tp = fp = fn = 0

    for i in range(n):
        for j in range(i + 1, n):
            true_same = true_labels[i] == true_labels[j]
            pred_same = pred_labels[i] == pred_labels[j]

            if true_same and pred_same:
                # true positives: pairs in same cluster in both true and predicted
                tp += 1
            elif pred_same and not true_same:
                # false positives: pairs in same predicted cluster, different true cluster
                fp += 1
            elif true_same and not pred_same:
                # false negatives: pairs in same true cluster, different predicted cluster
                fn += 1

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return np.round(f1, 4)


def evaluate_clustering(true_groups, predicted_groups):
    """
    Evaluate predicted clustering against ground truth.

    Parameters
    ----------
        true_groups: dict like {group_name: [item1, item2, ...]}
        predicted_groups: dict like {group_name: [item1, item2, ...]}

    Returns:
        dict of evaluation metrics:
        . adjusted_rand_index (ARI):
            - measures similarity of two grouping assignments
            - symmetric and robust to label permutation
            - Range: [-0.5, 1.0], adjusted for chance (1 = perfect match, 0 = random match)
            -> Use ARI when the ground truth clustering has large equal sized clusters

        . adjusted_mutual_info_score (AMI):
            - measures agreement of two grouping assignments
            - symmetric and robust to label permutation
            - Range: [-inf, 1.0], adjusted for chance (1 = perfect match, 0 = random match)
            -> Use AMI when the ground truth clustering is unbalanced and there exist small clusters
            ref: https://stats.stackexchange.com/questions/260487/adjusted-rand-index-vs-adjusted-mutual-information

        . homogeneity, completeness, V-Measure
            - homogeneity [0,1]: refers to cluster homogeneity: all clusters contain only items from a single class
            - completeness [0,1]: clusters are complete if they capture all items belong to one group (i.e., true groups are not split)
            - v-measure [0,1]: harmonic mean between homogeneity and completeness
            - all three are robust to label permutation, V-Measure is also symmetric

        . pairwise f1
            - measures accuracy of grouping via f1 of all pairs of items and whether they're grouped correctly
            - stricter than V-Measure
    """
    # get label arrays
    true_labels, pred_labels, _ = groups_to_labels(true_groups, predicted_groups)

    # compute metrics
    metrics = {
        "adjusted_rand_index": np.round(
            adjusted_rand_score(true_labels, pred_labels), 4
        ),
        "adjusted_mutual_info": np.round(
            adjusted_mutual_info_score(true_labels, pred_labels), 4
        ),
    }

    # homogeneity, completeness, V-Measure
    h, c, v = homogeneity_completeness_v_measure(true_labels, pred_labels)
    metrics["homogeneity"] = np.round(h, 4)
    metrics["completeness"] = np.round(c, 4)
    metrics["v_measure"] = np.round(v, 4)

    # pairwise f1
    metrics["pairwise_f1"] = compute_pairwise_f1(true_labels, pred_labels)

    return metrics
